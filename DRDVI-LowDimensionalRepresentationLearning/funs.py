import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class DiffusionDimReduction(nn.Module):
    def __init__(self, D, dev):
        super(DiffusionDimReduction, self).__init__()
        self.dev = dev
        self.D = D     # [d_0, d_1, ....,d_T, d_z, n]
        self.T = self.D.size(dim=0)-3  # number of layers

        self.sigma = nn.Parameter(torch.tensor([0.]))

        self.a = nn.Parameter(torch.linspace(-1., 1., self.T))

        self.A = nn.ParameterList(
            [nn.Parameter(
                torch.rand(self.D[ll], self.D[ll + 1]) * 2. * torch.sqrt(1. / self.D[ll + 1]) - torch.sqrt(
                    1. / self.D[ll + 1])) for ll in range(self.T)])

        self.B = nn.ParameterList(
            [nn.Parameter(
                torch.rand(self.D[ll + 1], self.D[ll]) * 2. * torch.sqrt(1. / self.D[ll]) - torch.sqrt(
                    1. / self.D[ll])) for ll in range(self.T)])

        self.Az = nn.Parameter(
            torch.rand(self.D[-3], self.D[-2]) * 2. * torch.sqrt(1. / self.D[-2]) - torch.sqrt(1. / self.D[-2]))

        self.encoder_a = nn.Sequential(
          nn.Linear(self.D[-3], self.D[-2], bias=False),
        )
        self.encoder_b = nn.Sequential(
          nn.Linear(self.D[-3], self.D[-2], bias=False),
        )


    def forward(self, input):
        # load data
        bs = input.size(0)  # number of input samples.
        x0 = input.view(bs, -1)
        x0 = x0.T           # row:features; column: data samples

        a_sig = self.act_fun(self.a, "sig")
        sig2 = torch.exp(self.sigma)

        # reconstruction term
        var_eps = torch.randn([self.D[1], bs], device=self.dev)
        a_acc = a_sig[0]
        B_acc = self.B[0]
        xt = torch.sqrt(a_acc)*B_acc@x0 + torch.sqrt(1.-a_acc)*var_eps
        mu_qx1 = self.A[0] @ xt

        rec_loss = torch.sum(torch.square(x0 - mu_qx1)) / bs * 0.5 / sig2 + self.D[0] * 0.5 * sig2

        # layer matching term
        layer_match_loss = torch.tensor([0.], device=self.dev)
        for tt in range(1, self.T):
            bar_a_t_1 = a_acc
            bar_B_t_1 = B_acc
            a_acc = a_sig[tt] * a_acc
            B_acc = self.B[tt] @ B_acc

            var_eps = torch.randn([self.D[tt+1], bs], device=self.dev)
            xt = torch.sqrt(a_sig[tt])*self.B[tt]@xt + torch.sqrt(1.-a_sig[tt])*var_eps
            temp_a = (1 - bar_a_t_1) / (1 - a_acc)

            xhat = self.act_fun(self.A[tt]@xt, "relu")
            diff = torch.sqrt(bar_a_t_1) * (xhat-bar_B_t_1 @ x0) - torch.sqrt(bar_a_t_1) * a_sig[tt] * temp_a * (
                        self.B[tt].T @ self.B[tt]) @ (xhat-bar_B_t_1 @ x0)

            temp = torch.sum(torch.square(diff)) / (1 - bar_a_t_1) + a_sig[tt] / (1 - a_sig[tt]) * torch.sum(
                torch.square(self.B[tt] @ diff))
            layer_match_loss = layer_match_loss + 0.5*temp/bs


        alpha = torch.exp( self.encoder_a(xt.T).T )
        beta = torch.exp( self.encoder_b(xt.T).T )
        mean = alpha/(alpha+beta)
        ls_p = torch.sum(torch.square(xt - self.Az@mean))
        cov_diag = (alpha*beta)/torch.square(alpha+beta)/(alpha+beta+1.)
        Cov = torch.diag(torch.sum(cov_diag, dim=1))
        trACAT = torch.trace(self.Az@Cov@self.Az.T)
        loss_xT = (ls_p + trACAT)*0.5/(1-a_acc)/bs


        temp1 = (alpha-1)*torch.special.digamma(alpha)
        temp2 = (beta-1)*torch.special.digamma(beta) 
        temp3 = (alpha+beta-2)*torch.special.digamma(alpha+beta)
        temp4 = torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)
        pri_loss = torch.sum(temp1+temp2-temp3-temp4)/bs

        loss = rec_loss + layer_match_loss + loss_xT + pri_loss

        ort_pen = torch.tensor([0.], device=self.dev)
        for tt in range(self.T):
            diff = self.B[tt]@self.B[tt].T-torch.eye(self.D[tt+1], device=self.dev)
            ort_pen = ort_pen + torch.sum(torch.square(diff))

        return loss, ort_pen


    def encoder(self, x):
        for tt in range(self.T):
            a = self.act_fun(self.a[tt], "sig")
            x = torch.sqrt(a)*self.B[tt]@x
        alpha = torch.exp( self.encoder_a(x.T).T )
        beta = torch.exp( self.encoder_b(x.T).T )
        return alpha/(alpha+beta)

    def act_fun(self, input, type):
        if type == "relu":
            output = nn.functional.relu(input)
        elif type == "sig":
            output = torch.sigmoid(input)
        elif type == "exp":
            output = torch.exp(input)
        else:
            output = input
        return output


def mat_to_Tensor(path: str, data_name: str, datatype):
    mat_dict = scio.loadmat(path)
    mat_data = mat_dict[data_name] / 1.0
    mat_data = torch.tensor(mat_data, dtype=datatype)
    return mat_data


def kmeans_clustering_accuracy(data_tensor, label_tensor, numbers=10):

    # Helper function to compute clustering accuracy using Hungarian algorithm
    def clustering_accuracy(y_true, y_pred):
        assert y_true.size == y_pred.size
        D = max(y_pred.max(), y_true.max()) + 1
        cost_matrix = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            cost_matrix[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_assignment(cost_matrix.max() - cost_matrix)
        accuracy = cost_matrix[row_ind, col_ind].sum() / y_pred.size
        return accuracy


    data_samples = data_tensor.T.cpu().numpy()  # (samples, features)
    labels = label_tensor.view(-1).cpu().numpy()  # (samples,)

    # Determine number of clusters from unique labels
    num_clusters = len(np.unique(labels))

    # Run KMeans multiple times and keep best accuracy
    kmeans = KMeans(n_clusters=num_clusters, n_init=numbers)
    predicted_clusters = kmeans.fit_predict(data_samples)
    acc = clustering_accuracy(labels, predicted_clusters)
    nmi = normalized_mutual_info_score(labels, predicted_clusters)
    ari = adjusted_rand_score(labels, predicted_clusters)

    return acc, nmi, ari