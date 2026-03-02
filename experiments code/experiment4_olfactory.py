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

# data can be downloaded from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148360
adata_sc = sc.read_h5ad("/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/olfactory/adata_sc.h5ad")
category_to_num = {category: idx for idx, category in enumerate(np.unique(adata_sc.obs['class']))}
label = np.array([category_to_num[category] for category in adata_sc.obs['class']])
adata_sc.obs['label'] = label
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(skf.split(adata_sc.X, adata_sc.obs["class"]))
adata_train = adata_sc[train_idx]
adata_val = adata_sc[test_idx]
#option1
cov = pd.read_csv('/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/olfactory/olfactory_update_cov.csv')
#coexp  = pd.read_csv('/gpfs/gibbs/project/zhao/yg399/gene_panel_design/dataset/olfactory/olfactory_coexp.csv')
epsilon = 1e-4 
cov = cov + epsilon * np.eye(cov.shape[0])
L = np.linalg.cholesky(np.matrix(cov))
L = torch.from_numpy(L).float()

from tqdm import tqdm
from persist import ExpressionDataset
train_dataset = ExpressionDataset(adata_train.X, adata_train.X)
val_dataset = ExpressionDataset(adata_val.X, adata_val.X)

device = "cpu"


selector_1 = scGPD(train_dataset,
                   val_dataset,
                   hidden = [128,128],
                   L = L.float(),
                   loss_fn=PoiLoss(),
                   device=device)
print('Starting')
candidates, _ = selector_1.eliminate(target=500,max_nepochs=150,lam_init=0.00000185,verbose=False)
print('Completed')

train_dataset = ExpressionDataset(adata_train[:,candidates].layers['log1pcpm'], adata_train.obs['label'])
val_dataset = ExpressionDataset(adata_val[:,candidates].layers['log1pcpm'], adata_val.obs['label'])

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











