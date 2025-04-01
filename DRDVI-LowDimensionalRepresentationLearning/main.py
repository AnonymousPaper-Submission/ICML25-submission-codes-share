import torch
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


path = "./"
data = funs.mat_to_Tensor(path+'cmupiedata.mat',"dt", torch.get_default_dtype()).T.to(dev)
lb = funs.mat_to_Tensor(path+'label.mat',"lb", torch.int).to(dev)

n_samples = data.size()[0]
data_dim = data.size()[1]
batch_size = int(n_samples/100)
epochsize = 501
D = torch.tensor([data_dim, 256, 128, 64, 32, 16, n_samples])

DRDVI = funs.DiffusionDimReduction(D, dev)
DRDVI = DRDVI.to(dev)
optimizer = optim.Adam(DRDVI.parameters(), lr=0.001, betas=(0.8, 0.888))
lambda_pen = torch.tensor([1e6], device=dev)
iter_per_epoch = int(np.ceil(n_samples/batch_size))
print("start train...")
for epoch in range(epochsize):
    idx_permu = torch.randperm(n_samples)
    # training
    for ii in range(iter_per_epoch):
        start_idx = ii*batch_size
        end_idx = min((ii + 1) * batch_size, n_samples - 1)
        batch_data = data[idx_permu[start_idx:end_idx],].to(dev)

        loss, orth = DRDVI(batch_data)
        loss_all = loss + lambda_pen*orth/(end_idx-start_idx)

        loss_all.backward()
        optimizer.step()
        optimizer.zero_grad()
        if ii % 200 == 0:
            print("epoch:{}/{}, loss:{}"
                  .format(epoch, epochsize, loss))

    if epoch % 100 == 0:
        with torch.no_grad():
            z = DRDVI.encoder(data.T.to(dev))
            acc, nmi, ari = funs.kmeans_clustering_accuracy(z, lb, 10)
            print("ACC:", acc, "NMI:", nmi, "ARI:", ari)
