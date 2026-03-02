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

# data can be downloaded from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078
adata = sc.read_h5ad("/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/breast_cancer/all_raw_data.h5ad")
sc.pp.highly_variable_genes(adata_sc, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)
sc.tl.rank_genes_groups(adata_sc, 'celltype_major', method='wilcoxon')
DE_df = sc.get.rank_genes_groups_df(adata_sc, group=None, log2fc_min=4)
DE_df = DE_df[DE_df.pvals_adj<0.05]
DE_df = DE_df[DE_df.names.isin(adata.var.index)]
adata_hvg = adata_sc[:,DE_df['names']]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(skf.split(adata.X, adata.obs["celltype_major"]))
adata_train = adata[train_idx]
adata_val = adata[test_idx]
category_to_num = {category: idx for idx, category in enumerate(np.unique(adata.obs['celltype_major']))}
label = np.array([category_to_num[category] for category in adata.obs['celltype_major']])
adata.obs['label'] = label
cov = pd.read_csv('/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/breast_cancer/breast_cancer_cov.csv')
epsilon = 1e-5 
cov = cov + epsilon * np.eye(cov.shape[0])
L = np.linalg.cholesky(np.matrix(cov))
L = torch.from_numpy(L).float()
from tqdm import tqdm
train_dataset = ExpressionDataset(adata_train.X.toarray(), adata_train.X.toarray())
val_dataset = ExpressionDataset(adata_val.X.toarray(), adata_val.X.toarray())
device = torch.device('cpu')
selector_1 = scGPD(train_dataset,
                   val_dataset,
                   hidden = [128,128],
                   L = L.float(),
                   loss_fn=PoiLoss(),
                   device=device)
candidates, _ = selector_1.eliminate(target=800,max_nepochs=150,lam_init=0.000027,verbose=False)

train_dataset = ExpressionDataset(adata_train.layers['log1pcpm'][:,candidates], adata_train.obs['label'].values)
val_dataset = ExpressionDataset(adata_val.layers['log1pcpm'][:,candidates], adata_val.obs['label'].values)
num_genes_list = [256,128,64,32]
scGPD_result = {}
selector = scGPD(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   L = L.float(),
                   device=device)

for num in num_genes_list:
    inds, model = selector.select(num_genes=num, max_nepochs=280,verbose=True)
    scGPD_result[num] = inds




