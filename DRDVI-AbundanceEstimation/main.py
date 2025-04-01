import torch
import torchvision
import torch.optim as optim
import numpy as np
import funs
import random


# Uncomment this to use higher precision
#torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
dev = torch.device("cpu")

# load data
datapath = "./jasperRidge2_R198"
refpath = "./end4"
data = funs.mat_to_Tensor(datapath,"Y", datatype=torch.get_default_dtype()).T
nCol = funs.mat_to_Tensor(datapath,"nCol", torch.int).view(1)
nRow = funs.mat_to_Tensor(datapath,"nRow", torch.int).view(1)
n_samples = data.size()[0]
data_dim = data.size()[1]
A_ref = funs.mat_to_Tensor(refpath,"M", torch.get_default_dtype())
Z_ref = funs.mat_to_Tensor(refpath,"A", torch.get_default_dtype()).to(dev)
end_num = A_ref.size()[1]

abundance_ref = torch.transpose(Z_ref.view(end_num , 1, nCol, nRow), 2, 3)
torchvision.utils.save_image(abundance_ref, "abundance_ref.jpg", nrow=end_num , normalize=True, scale_each=True)



batch_size = int(n_samples/100)
epochsize = 501

D = torch.tensor([data_dim, 64, 32, 16, 8, end_num, n_samples])


DRDVI = funs.DiffusionDimReduction(D, dev)
DRDVI = DRDVI.to(dev)
iter_per_epoch = int(np.ceil(n_samples / batch_size))
optimizer = optim.Adam(DRDVI.parameters(), lr=0.001, betas=(0.8, 0.888))
lambda_pen = torch.tensor([100000.], device=dev)
print("start train...")
for epoch in range(epochsize):
    idx_permu = torch.randperm(n_samples)
    for ii in range(iter_per_epoch):
        start_idx = ii*batch_size
        end_idx = min([(ii+1)*batch_size, n_samples-1])
        batch_data = data[idx_permu[start_idx:end_idx],].to(dev)

        loss, orth = DRDVI(batch_data)
        loss_all = loss + lambda_pen*orth/(end_idx-start_idx)

        loss_all.backward()
        optimizer.step()
        optimizer.zero_grad()
        if ii % 200 == 0:
            print("epoch:{}/{}, loss:{}"
                  .format(epoch, epochsize, loss))

    with torch.no_grad():
        loss,_ = DRDVI(data.to(dev))
        z = DRDVI.encoder(data.T.to(dev))
        # comment to speed up
        z, MSE = funs.align_and_compute_rowwise_mse(Z_ref, z)
        print('Average MSE: ', MSE/end_num)
        abundance_est = torch.transpose(z.view(D[-2], 1, nCol, nRow), 2, 3)
        torchvision.utils.save_image(abundance_est, "est_abundance" + ".jpg", nrow=D[-2],
                                    normalize=True, scale_each=True)