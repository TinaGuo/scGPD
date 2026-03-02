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
# read data
# data can be downloaded from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907

df = pd.read_csv("/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/lung_counts.csv", index_col=0)
adata = sc.AnnData(df)
from scipy.sparse import csr_matrix
adata.X = csr_matrix(adata.X)
adata.layers['log1pcpm'] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
sc.pp.log1p(adata, layer='log1pcpm')
adata.obs["condition"] = np.where(adata.obs_names.str.contains("LUNG_N18"), 0, 1)
sc.pp.highly_variable_genes(adata, 
                            layer='log1pcpm', 
                            flavor='cell_ranger',
                            n_top_genes=5000, 
                            inplace=True)
adata_hvg = adata[:, adata.var["highly_variable"]]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(skf.split(adata_hvg.X, adata_hvg.obs["condition"]))
device = torch.device('cpu')
adata_train = adata_hvg[train_idx]
adata_val = adata_hvg[test_idx]

cov = pd.read_csv(
    "/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/lung_cov.csv",
    index_col=0
)
epsilon = 1e-6
cov = cov + epsilon * np.eye(cov.shape[0])
L = np.linalg.cholesky(cov.values)
L = torch.from_numpy(L).float()

train_dataset = ExpressionDataset(adata_train.X, adata_train.X)
val_dataset = ExpressionDataset(adata_val.X, adata_val.X)
selector_1 = scGPD(train_dataset,
                   val_dataset,
                   hidden = [128,128],
                   L = L.float(),
                   loss_fn=PoiLoss(),
                   device=device)
print('Starting')
candidates, _ = selector_1.eliminate(target=500,max_nepochs=150,lam_init=0.000047,verbose=False)
print('Completed')

train_dataset = ExpressionDataset(adata_train.layers['log1pcpm'], adata_train.obs['condition'].values)
val_dataset = ExpressionDataset(adata_val.layers['log1pcpm'], adata_val.obs['condition'].values)


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


