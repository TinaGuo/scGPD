import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scGPD import scGPD, ExpressionDataset,PoiLoss


## load data
# data can be downloaded from: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k
adata = sc.read_h5ad("/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/single_cell/pbmc_preprocessed.h5ad")
adata.obs["louvain_id"] = adata.obs["louvain"].cat.codes

cov = pd.read_csv('/home/yg399/cs_core/pbmc_cov.csv')
coexp  = pd.read_csv('/home/yg399/cs_core/pbmc_coexp.csv')
epsilon = 1e-6 
cov = cov + epsilon * np.eye(cov.shape[0])
L = np.linalg.cholesky(np.matrix(cov))
L = torch.from_numpy(L).float()
device = torch.device('cpu')

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(skf.split(adata.X, adata.obs["louvain_id"]))
adata_train = adata[train_idx]
adata_val = adata[test_idx]
train_dataset = ExpressionDataset(adata_train.X, adata_train.X)
val_dataset   = ExpressionDataset(adata_val.X,   adata_val.X)
selector_1 = scGPD(train_dataset,
                   val_dataset,
                   hidden = [128,128],
                   L = L.float(),
                   loss_fn=PoiLoss(),
                   device=device)
print('Starting')
candidates, _ = selector_1.eliminate(target=500,max_nepochs=150,lam_init=0.0000357,verbose=False)
print('Completed')

train_dataset = ExpressionDataset(adata_train.layers['log1pcpm'][:,candidates], adata_train.obs['louvain_id'].values)
val_dataset = ExpressionDataset(adata_val.layers['log1pcpm'][:,candidates], adata_val.obs['louvain_id'].values)

num_genes_list = [256,128,64,32]
scGPD_result = {}

selector = scGPD(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   L = L.float(),
                   device=device)

for num in num_genes_list:
    inds, model = selector.select(num_genes=num, max_nepochs=250,verbose=True)
    scGPD_result[num] = inds









