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

# data can be downloaded from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133

adata = sc.read_h5ad("/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/single_cell/human_hvg.h5ad")

adata = adata[:,adata.var['highly_variable']]

category_to_num = {category: idx for idx, category in enumerate(np.unique(adata.obs['cell_type1']))}

label = np.array([category_to_num[category] for category in adata.obs['cell_type1']])

adata.obs['label'] = label
from scipy.sparse import csr_matrix
adata.X = csr_matrix(adata.X)
adata.layers['log1pcpm'] = csr_matrix(adata.layers['log1pcpm'])

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(skf.split(adata.X, adata.obs["label"]))

adata_train = adata[train_idx]
adata_val = adata[test_idx]

cov = pd.read_csv(
    "/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/single_cell/human_cov.csv",
    index_col=0
)
epsilon = 1e-5
cov = cov + epsilon * np.eye(cov.shape[0])

L = np.linalg.cholesky(cov.values)

L = torch.from_numpy(L).float()

from tqdm import tqdm  
# Build PyTorch datasets for scGPD
train_dataset = ExpressionDataset(adata_train.X, adata_train.X)
val_dataset   = ExpressionDataset(adata_val.X,   adata_val.X)
device = "cpu"

print(f"Using device: {device}")
# step 1 
selector_1 = scGPD(train_dataset,
                   val_dataset,
                   hidden = [128,128],
                   L = L.float(),
                   loss_fn=PoiLoss(),
                   device=device)

print('Starting')
candidates, _ = selector_1.eliminate(target=500,max_nepochs=150,lam_init=0.0000357,verbose=False)
print('Completed')


train_dataset = ExpressionDataset(adata_train.layers['log1pcpm'][:,candidates], adata_train.obs['label'].values)
val_dataset = ExpressionDataset(adata_val.layers['log1pcpm'][:,candidates], adata_val.obs['label'].values)

num_genes_list = [256,128,64,32]
scGPD_result = {}

selector = scGPD(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   L = L.float(),
                   device=device)

print('Selecting genes')
for num in num_genes_list:
    inds, model = selector.select(num_genes=num, max_nepochs=280,verbose=True)
    scGPD_result[num] = inds
print('Done')

