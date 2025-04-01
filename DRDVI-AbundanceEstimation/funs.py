import torch
import torch.nn as nn
import math
from scipy.optimize import linear_sum_assignment
import scipy.io as scio


class DiffusionDimReduction(nn.Module):
    def __init__(self, D, dev):
        super(DiffusionDimReduction, self).__init__()
        self.dev = dev
        self.D = D  # [d_0, d_1, ..., d_T, d_z, n]
        self.T = len(self.D) - 3  # number of layers
        assert self.T > 0, "The number of layers (T) must be greater than zero!"

        # Trainable parameters
        self.sigma = nn.Parameter(torch.zeros(1, device=self.dev))  # Scalar parameter
        self.a = nn.Parameter(torch.linspace(-1.0, 1.0, self.T, device=self.dev))  # Layer-wise scaling factors

        # Layer weight matrices
        self.A = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.D[ll], self.D[ll + 1], device=self.dev)
            ) for ll in range(self.T)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.D[ll + 1], self.D[ll], device=self.dev)
            ) for ll in range(self.T)
        ])

        # Initialize layer weights
        for param in self.A:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in self.B:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))

        # top layer weights
        self.Az = nn.Parameter(torch.empty(self.D[-3], self.D[-2], device=self.dev))
        self.Bz = nn.Parameter(torch.empty(self.D[-2], self.D[-3], device=self.dev))
        nn.init.kaiming_uniform_(self.Az, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Bz, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input (torch.Tensor): Input data tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and orthogonality penalty.
        """
        bs = input.size(0)  # Batch size
        x0 = input.view(bs, -1).T  # Transpose: rows=features, cols=samples

        # Compute activations and scaling
        a_sig = self.act_fun(self.a, "sig")
        sig2 = torch.exp(self.sigma)

        # Reconstruction term
        var_eps = torch.randn(self.D[1], bs, device=self.dev)
        a_acc = a_sig[0]
        B_acc = self.B[0]
        xt = torch.sqrt(a_acc) * B_acc @ x0 + torch.sqrt(1.0 - a_acc) * var_eps
        mu_qx1 = self.act_fun(self.A[0] @ xt, "relu")
        rec_loss = (torch.sum((x0 - mu_qx1) ** 2) / bs) * 0.5 / sig2 + self.D[0] * 0.5 * sig2

        # Layer matching term
        layer_match_loss = torch.zeros(1, device=self.dev)
        for tt in range(1, self.T):
            # Update accumulations
            bar_a_t_1 = a_acc
            bar_B_t_1 = B_acc
            a_acc = a_sig[tt] * a_acc
            B_acc = self.B[tt] @ B_acc

            # Intermediate computations
            var_eps = torch.randn(self.D[tt + 1], bs, device=self.dev)
            xt = torch.sqrt(a_sig[tt]) * self.B[tt] @ xt + torch.sqrt(1.0 - a_sig[tt]) * var_eps
            temp_a = (1 - bar_a_t_1) / (1 - a_acc)

            # Layer matching loss
            xhat = self.act_fun(self.A[tt] @ xt, "relu")
            diff = (
                torch.sqrt(bar_a_t_1) * (xhat - bar_B_t_1 @ x0)
                - torch.sqrt(bar_a_t_1) * a_sig[tt] * temp_a
                * (self.B[tt].T @ self.B[tt]) @ (xhat - bar_B_t_1 @ x0)
            )
            temp = (
                torch.sum(diff ** 2) / (1 - bar_a_t_1)
                + a_sig[tt] / (1 - a_sig[tt]) * torch.sum((self.B[tt] @ diff) ** 2)
            )
            layer_match_loss += 0.5 * temp / bs


        eta = self.act_fun(self.Bz @ xt, "exp")
        tilde_eta = eta.sum(dim=0, keepdim=True)
        eta_bar = eta / tilde_eta
        ls_p = torch.sum((xt - self.Az @ eta_bar) ** 2)
        Cov = torch.diag(torch.sum(eta_bar / (1 + tilde_eta), dim=1)) - (eta_bar / (1 + tilde_eta)) @ eta_bar.T
        trACAT = torch.trace(self.Az @ Cov @ self.Az.T)
        eps = 1e-12
        loss_xT = (ls_p + trACAT) * 0.5 / (1 - a_acc + eps) / bs


        diaga_eta = torch.special.digamma(eta)
        diaga_tilde_eta = torch.special.digamma(tilde_eta)
        lnga_eta = torch.special.gammaln(eta)
        lnga_tilde_eta = torch.special.gammaln(tilde_eta)
        pri_loss = (
            torch.sum((self.D[-2] - tilde_eta) * diaga_tilde_eta)
            + torch.sum((eta - 1) * diaga_eta)
            + torch.sum(lnga_tilde_eta)
            - torch.sum(lnga_eta)
        ) / bs

        loss = rec_loss + layer_match_loss + loss_xT + pri_loss

        ort_pen = torch.zeros(1, device=self.dev)
        for tt in range(self.T):
            diff = self.B[tt] @ self.B[tt].T - torch.eye(self.B[tt].size(0), device=self.dev)
            ort_pen += torch.sum(diff ** 2)

        return loss, ort_pen

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded representation.
        """
        for tt in range(self.T):
            a = self.act_fun(self.a[tt], "sig")
            x = torch.sqrt(a) * self.B[tt] @ x
        x = self.act_fun(self.Bz @ x, "exp")
        return x/x.sum(dim=0, keepdim=True)

    def act_fun(self, input: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Applies activation functions.

        Args:
            input (torch.Tensor): Input tensor.
            mode (str): Type of activation function.

        Returns:
            torch.Tensor: Activated tensor.
        """
        if mode == "relu":
            output = nn.functional.relu(input)
        elif mode == "sig":
            output = torch.sigmoid(input)
        elif mode == "exp":
            output = torch.exp(input)
        else:
            output = input
        return output


def align_and_compute_rowwise_mse(x1: torch.Tensor, x2: torch.Tensor, distance_metric='euclidean') -> torch.Tensor:
    """
    Align rows of tensor x2 to x1 using Hungarian algorithm, then compute mse

    Args:
        x1 (torch.Tensor): Tensor of shape (N, D).
        x2 (torch.Tensor): Tensor of shape (N, D).
        distance_metric (str): Distance metric for alignment ('euclidean', 'cosine').

    """
    assert x1.shape == x2.shape, "Input tensors must have the same shape"

    # Compute distance matrix for assignment
    if distance_metric == 'euclidean':
        dist_matrix = torch.cdist(x1, x2, p=2)  # (N, N)
    elif distance_metric == 'cosine':
        x1_norm = torch.nn.functional.normalize(x1, dim=1)
        x2_norm = torch.nn.functional.normalize(x2, dim=1)
        similarity_matrix = torch.mm(x1_norm, x2_norm.T)
        dist_matrix = 1 - similarity_matrix  # cosine distance
    else:
        raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'cosine'.")

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

    # Align x2
    x2_aligned = x2[col_ind]

    # Compute row-wise normalized errors
    row_diff_norm = torch.norm(x1 - x2_aligned, dim=1)
    x1_row_norm = torch.norm(x1, dim=1)

    # Handle division by zero if any row in x1 is zero
    rowwise_mse = row_diff_norm / x1_row_norm

    return  x2_aligned, torch.sum(rowwise_mse)



def mat_to_Tensor(path: str, data_name: str, datatype):
    mat_dict = scio.loadmat(path)
    mat_data = mat_dict[data_name]/1.0
    mat_data = torch.tensor(mat_data, dtype=datatype)
    return mat_data