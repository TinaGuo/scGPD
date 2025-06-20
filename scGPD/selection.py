import torch
import numpy as np
import torch.nn as nn
from scGPD import functions, utils


class scGPD:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 loss_fn,
                 device,
                 L,
                 prior_inds = [],
                 preselected_inds=[],
                 hidden=[128, 128],
                 activation=nn.ReLU(),seed=123):
        # TODO add verification for dataset type.
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        
        self.train = train_dataset
        self.val = val_dataset
        self.loss_fn = loss_fn

        # Architecture parameters.
        self.hidden = hidden
        self.activation = activation
        self.L = L
        self.prior_inds = np.sort(prior_inds).astype(int)

        # Set device.
        assert isinstance(device, torch.device)
        self.device = device

        # Set preselected genes.
        self.preselected = np.sort(preselected_inds).astype(int)

        # Initialize candidate genes.
        self.set_genes()

    def get_genes(self):
        '''Get currently selected genes, not including preselected genes.'''
        return self.candidates

    def set_genes(self, candidates=None):
        '''Restrict the subset of genes.'''
        if candidates is None:
            # All genes but pre-selected ones.
            candidates = np.array(
                [i for i in range(self.train.max_input_size)
                 if i not in self.preselected])
        
        else:
            # Ensure that candidates do not overlap with pre-selected genes.
            assert len(np.intersect1d(candidates, self.preselected)) == 0
        self.candidates = candidates

        # Set genes in datasets.
        included = np.sort(np.concatenate([candidates, self.preselected]))
        self.train.set_inds(included)
        self.val.set_inds(included)

        # Set relative indices for pre-selected genes.
        self.preselected_relative = np.array(
            [np.where(included == ind)[0][0] for ind in self.preselected])

    def eliminate(self,
                  target,
                  lam_init=None,
                  mbsize=64,
                  max_nepochs=250,
                  lr=1e-3,
                  tol=0.2,
                  start_temperature=10.0,
                  end_temperature=0.01,
                  optimizer='Adam',
                  lookback=10,
                  max_trials=10,
                  bar=True,
                  verbose=False):

        # Reset candidate genes.
        all_inds = np.arange(self.train.max_input_size)
        all_candidates = np.array_equal(
            self.candidates, np.setdiff1d(all_inds, self.preselected))
        all_train_inds = np.array_equal(self.train.inds, all_inds)
        all_val_inds = np.array_equal(self.val.inds, all_inds)
        if not (all_candidates and all_train_inds and all_val_inds):
            print('resetting candidate genes')
            self.set_genes()

        # Initialize architecture.
        if isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.MSELoss)):
            output_size = self.train.output_size
        else:
            output_size = self.train.output_size
            #print(f'Unknown loss function, assuming {self.loss_fn} requires '
             #     f'{self.train.output_size} outputs')

        model = functions.SelectorMLP(input_layer='binary_gates',
                                   input_size=self.train.input_size,
                                   output_size=output_size,
                                   hidden=self.hidden,
                                   L = self.L,
                                   activation=self.activation,
                                   preselected_inds=self.preselected_relative)
        model = model.to(self.device)

        # Determine lam_init, if necessary.
        if lam_init is None:
            print('unknown loss function, starting with lam = 0.0001')
            lam_init = 0.0001
        else:
            print(f'trying lam = {lam_init:.6f}')

        # Prepare for training and lambda search.
        assert 0 < target < self.train.input_size
        assert 0.1 <= tol < 0.5
        assert lam_init > 0
        lam_list = [0]
        num_remaining = self.train.input_size
        num_remaining_list = [num_remaining]
        lam = lam_init

        model.fit(self.train,
                      self.val,
                      lr,
                      mbsize,
                      max_nepochs,
                      start_temperature=start_temperature,
                      end_temperature=end_temperature,
                      loss_fn=self.loss_fn,
                      lam=lam,
                      optimizer=optimizer,
                      lookback=lookback,
                      bar=bar,
                      verbose=verbose)

            # Extract inds.
        inds = model.input_layer.get_inds(threshold=0.5)
        num_remaining = len(inds)
        print(f'lam = {lam:.6f} yielded {num_remaining} genes')


        # Set eligible genes.
        true_inds = self.candidates[inds]
        self.set_genes(true_inds)
        return true_inds, model

    def select(self,
               num_genes,
               mbsize=64,
               max_nepochs=250,
               lr=1e-3,
               start_temperature=10.0,
               end_temperature=0.01,
               optimizer='Adam',
               bar=True,
               verbose=False):

        # Possibly reset candidate genes.
        included_inds = np.sort(
            np.concatenate([self.candidates, self.preselected]))
        candidate_train_inds = np.array_equal(self.train.inds, included_inds)
        candidate_val_inds = np.array_equal(self.val.inds, included_inds)
        if not (candidate_train_inds and candidate_val_inds):
            print('setting candidate genes in datasets')
            self.set_genes(self.candidates)

        # Initialize architecture.
        output_size = self.train.output_size
        print(f'assuming loss function {self.loss_fn} requires '
                  f'{self.train.output_size} outputs')

        input_size = len(self.candidates) + len(self.preselected)
        model = functions.SelectorMLP_2(input_layer='binary_mask',
                                   input_size=input_size,
                                   output_size=output_size,
                                   hidden=[32,32],#self.hidden
                                   activation=self.activation,
                                   preselected_inds=self.preselected_relative,
                                   num_selections=num_genes).to(self.device)

        # Train.
        model.fit(self.train,
                  self.val,
                  lr,
                  mbsize,
                  max_nepochs,
                  start_temperature,
                  end_temperature,
                  loss_fn=self.loss_fn,
                  optimizer=optimizer,
                  bar=bar,
                  verbose=verbose)

        # Return genes.
        inds = model.input_layer.get_inds()
        true_inds = self.candidates[inds]
        print(f'done, selected {len(inds)} genes')
        return true_inds, model


def modified_secant_method(x0, y0, y1, x, y):

    # Get robust slope estimate.
    weights = 1 / np.abs(x - x0)
    slope = (
        np.sum(weights * (x - x0) * (y - y0)) /
        np.sum(weights * (x - x0) ** 2))

    # Clip slope to minimum value.
    slope = np.clip(slope, a_min=1e-6, a_max=None)

    # Guess x1.
    x1 = x0 + (y1 - y0) / slope
    return x1
