import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def clamp_probs(probs):
    '''Clamp probabilities to ensure stable logs.'''
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1-eps)



def concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for Concrete distribution (see eq. 10 of Maddison et al., 2017).

    Args:
      logits: Concrete logits parameters.
      temperature: Concrete temperature parameter.
      shape: sample shape.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    gumbels = - torch.log(- torch.log(u))
    scores = (logits + gumbels) / temperature
    return scores.softmax(dim=-1)


def bernoulli_concrete_sample(logits, temperature, L, shape=torch.Size([])):
    '''
    Sampling for BinConcrete distribution (see PyTorch source code, differs
    slightly from eq. 16 of Maddison et al., 2017).

    Args:
       logits: tensor of BinConcrete logits parameters.
       temperature: BinConcrete temperature parameter.
       shape: sample shape.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    z = torch.rand(uniform_shape)
    u = z @ L.t()
    u = torch.distributions.Normal(0, 1).cdf(u)
    u = clamp_probs(u)
    return torch.sigmoid((F.logsigmoid(logits) - F.logsigmoid(-logits)
                          + torch.log(u) - torch.log(1 - u)) / temperature)


class BinaryMask(nn.Module):
    '''
    Input layer that selects features by learning a k-hot mask.

    Args:
      input_size: number of inputs.
      num_selections: number of features to select.
      temperature: temperature for Concrete samples.
      gamma: used to map learned parameters to logits (helps convergence speed).
    '''

    def __init__(self,
                 input_size,
                 num_selections,
                 temperature=10.0,
                 gamma=1/3):
        super().__init__()
        self._logits = nn.Parameter(
            torch.zeros(num_selections, input_size, dtype=torch.float32,
                        requires_grad=True))
        self.input_size = input_size
        self.num_selections = num_selections
        self.output_size = input_size
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma

    @property
    def probs(self):
        return (self.logits).softmax(dim=1)

    def sample(self, n_samples):
        '''Sample approximate k-hot vectors.'''
        samples = concrete_sample(
            self.logits, self.temperature, torch.Size([n_samples]))
        return torch.max(samples, dim=-2).values

    def forward(self, x):
        '''Sample and apply mask.'''
        m = self.sample(len(x))
        x = x * m
        return x, m

    def get_inds(self):
        '''Get selected indices.'''
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return (f'input_size={self.input_size}, temperature={self.temperature},'
                f' num_selections={self.num_selections}')


class BinaryGates(nn.Module):
    '''
    Input layer that selects features by learning binary gates for each feature,
    similar to [1].

    [1] Dropout Feature Ranking for Deep Learning Models (Chang et al., 2017)

    Args:
      input_size: number of inputs.
      temperature: temperature for BinConcrete samples.
      init: initial value for each gate's probability of being 1.
      gamma: used to map learned parameters to logits (helps convergence speed).
    '''

    def __init__(self,
                 input_size,
                 temperature=0.1,
                 init=0.9,
                 gamma=1/2):
        super().__init__()
        init_logit = - torch.log(1 / torch.tensor(init) - 1) * gamma
        self._logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.output_size = input_size
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma
    
    @property
    def probs(self):
        return torch.sigmoid(self.logits)

    def sample(self, n_samples,L):
        '''Sample approximate binary masks.'''
        return bernoulli_concrete_sample(
            self.logits, self.temperature,L,torch.Size([n_samples]))

    def forward(self,x,L):
        '''Sample and apply mask.'''
        m = self.sample(len(x),L)
        x = x * m
        return x, m

    def get_inds(self, num_features=None, threshold=None):
        '''
        Get selected indices.

        Args:
          num_features: number of top features to return.
          threshold: probability threshold for determining selected features.
        '''
        if (num_features is None) == (threshold is None):
            raise ValueError('exactly one of num_features and threshold must be'
                             ' specified')

        if num_features:
            inds = torch.argsort(self.probs)[-num_features:]
        elif threshold:
            inds = (self.probs > threshold).nonzero()[:, 0]
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return f'input_size={self.input_size}, temperature={self.temperature}'

class ConcreteSelector(nn.Module):

    def __init__(self,
                 input_size,
                 num_selections,
                 temperature=10.0,
                 gamma=1/3):
        super().__init__()
        self._logits = nn.Parameter(
            torch.zeros(num_selections, input_size, dtype=torch.float32,
                        requires_grad=True))
        self.input_size = input_size
        self.num_selections = num_selections
        self.output_size = num_selections
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma

    @property
    def probs(self):
        return self.logits.softmax(dim=1)

    def sample(self, n_samples):
        '''Sample approximate binary matrices.'''
        return concrete_sample(
            self.logits, self.temperature, torch.Size([n_samples]))

    def forward(self, x):
        '''Sample and apply selector matrix.'''
        M = self.sample(len(x))
        x = torch.matmul(M, x.unsqueeze(2)).squeeze(2)
        return x, M

    def get_inds(self):
        '''Get selected indices.'''
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return (f'input_size={self.input_size}, temperature={self.temperature},'
                f' num_selections={self.num_selections}')


class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU()):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation function.
        self.activation = activation
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)
            x = self.dropout(x)
        r = self.fc[-1](x)
        r = torch.exp(r)

        return r #self.fc[-1](x) #r

    def fit(self,
            train_dataset,
            val_dataset,
            mbsize,
            max_nepochs,
            loss_fn,
            lr=1e-3,
            min_lr=1e-5,
            lr_factor=0.5,
            optimizer='Adam',
            lookback=10,
            bar=False,
            verbose=True):

        # Set up optimizer.
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)

        # Set up data loaders.
        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(
            train_dataset, batch_size=mbsize, shuffle=True, drop_last=True,
            worker_init_fn=train_init, num_workers=4)
        val_loader = DataLoader(
            val_dataset, batch_size=mbsize, worker_init_fn=val_init,
            num_workers=4)

        # Determine device.
        device = next(self.parameters()).device

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []
        best_model = None
        best_loss = np.inf
        best_epoch = None

        # Bar setup.
        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=False)

        # Begin training.
        for epoch in range(max_nepochs):
            # For tracking mean train loss.
            train_loss = 0
            N = 0

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Forward pass.
                pred = self.forward(x)

                # Calculate loss.
                loss = loss_fn(pred, y)

                # Update mean train loss.
                train_loss = (
                    (N * train_loss + mbsize * loss.item()) / (N + mbsize))
                N += mbsize

                # Gradient step.
                loss.backward()
                optimizer.step()
                self.zero_grad()

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.eval()
                val_loss = self.validate(val_loader, loss_fn).item()
                self.train()

                # Update learning rate.
                scheduler.step(val_loss)

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)
                #scheduler.step(val_loss)

                if verbose:
                    print(f'{"-" * 8}Epoch = {epoch + 1}{"-" * 8}')
                    print(f'Train loss = {train_loss:.4f}')
                    print(f'Val loss = {val_loss:.4f}')

            # Update bar.
            if bar:
                tqdm_bar.update(1)

            # Check for early stopping.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self)
                best_epoch = epoch
            elif (epoch - best_epoch) == lookback:
                # Skip bar to end.
                if bar:
                    tqdm_bar.n = max_nepochs

                if verbose:
                    print('Stopping early')
                break

        # Restore model parameters.
        restore_parameters(self, best_model)

    def validate(self, loader, loss_fn):
        '''Calculate average loss.'''
        device = next(self.parameters()).device
        mean_loss = 0
        N = 0

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device=device)
                y = y.to(device=device)
                n = len(x)

                # Calculate loss.
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                mean_loss = (N * mean_loss + n * loss) / (N + n)
                N += n

        return mean_loss


class SelectorMLP(nn.Module):

    def __init__(self,
                 input_layer,
                 input_size,
                 output_size,
                 hidden,
                 L,
                 activation=nn.ReLU(),
                 preselected_inds=[],
                 prior_inds= [],
                 num_selections=None,
                 **kwargs):
        # Verify arguments.
        super().__init__()
        if num_selections is None:
            if input_layer in ('binary_mask', 'concrete_selector'):
                raise ValueError(
                    f'must specify num_selections for {input_layer} layer')
        else:
            if input_layer in ('binary_gates'):
                raise ValueError('num_selections cannot be specified for '
                                 f'{input_layer} layer')

        # Set up for pre-selected features.
        preselected_inds = np.sort(preselected_inds)
        assert len(preselected_inds) < input_size
        self.preselected = np.array(
            [i in preselected_inds for i in range(input_size)])
        preselected_size = len(preselected_inds)
        self.has_preselected = preselected_size > 0
        self.layer_name = input_layer
        
        prior_inds = np.sort(prior_inds)
        self.prior_inds = np.array(
            [i in prior_inds for i in range(input_size)])

        # Set up input layer.
        if input_layer == 'binary_mask':
            mlp_input_size = input_size
            self.input_layer = BinaryMask(
                input_size - preselected_size, num_selections, **kwargs)
        elif input_layer == 'binary_gates':
            mlp_input_size = input_size
            self.input_layer = BinaryGates(
                input_size - preselected_size, **kwargs)
        elif input_layer == 'concrete_selector':
            mlp_input_size = num_selections + preselected_size
            self.input_layer = layers.ConcreteSelector(
                input_size - preselected_size, num_selections, **kwargs)
        else:
            raise ValueError('unsupported input layer: {}'.format(input_layer))

        # Create MLP.
        self.mlp = MLP(mlp_input_size, output_size, hidden, activation)
        self.L = L

    def forward(self, x):
        '''Apply input layer and return MLP output.'''
        if self.has_preselected:
            pre = x[:, self.preselected]
            x, m = self.input_layer(x[:, ~self.preselected],self.L)
            x = torch.cat([pre, x], dim=1)
        else:
            if self.layer_name == 'binary_gates':
                x, m = self.input_layer(x,self.L)
            else:
                x, m = self.input_layer(x)       
        pred = self.mlp(x)
        return pred, x, m

    def fit(self,
            train_dataset,
            val_dataset,
            lr,
            mbsize,
            max_nepochs,
            start_temperature,
            end_temperature,
            loss_fn,
            eta=0,
            lam=0,
            beta=0,
            optimizer='Adam',
            lookback=10,
            bar=False,
            verbose=True):
        '''
        Train the model.

        Args:
          train_dataset: training dataset.
          val_dataset: validation dataset.
          lr: learning rate.
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          start_temperature:
          end_temperature:
          loss_fn: loss function.
          eta: penalty parameter for number of expressed genes.
          lam: penalty parameter.
          optimizer: optimizer type.
          lookback: number of epochs to wait for improvement before stopping.
          bar: whether to display tqdm progress bar.
          verbose: verbosity.
        '''
        # Verify arguments.
        if lam != 0:
            if not isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam should only be specified when using '
                                 'BinaryGates layer')
        else:
            if isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam must be specified when using '
                                 'BinaryGates layer')
        if eta > 0:
            if isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam cannot be specified when using '
                                 'BinaryGates layer')

        if end_temperature > start_temperature:
            raise ValueError('temperature should be annealed downwards, must '
                             'have end_temperature <= start_temperature')
        elif end_temperature == start_temperature:
            loss_early_stopping = True
        else:
            loss_early_stopping = False

        # Set up optimizer.
        optimizer = optimizer = optim.Adam(self.parameters(), lr=lr)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,35,50,70,90], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, factor=0.5, patience=lookback // 2, min_lr=1e-5,
            verbose=verbose)
 

        # Set up data loaders.
        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(train_dataset, batch_size=mbsize,
                                  shuffle=True, drop_last=True,
                                  worker_init_fn=train_init, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=mbsize,
                                worker_init_fn=val_init, num_workers=4)

        # Determine device.
        device = next(self.parameters()).device

        # Set temperature and determine rate for decreasing.
        self.input_layer.temperature = start_temperature
        r = np.power(end_temperature / start_temperature,
                     1 / ((len(train_dataset) // mbsize) * max_nepochs))

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []
        best_loss = np.inf
        best_epoch = -1

        # Bar setup.
        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=False)

        # Begin training.
        for epoch in range(max_nepochs):
            # For tracking mean train loss.
            train_loss = 0
            N = 0

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Calculate loss.
                pred, x, m = self.forward(x)
                loss = loss_fn(pred, y)

                # Calculate penalty if necessary.
                if lam > 0:
                    penalty = input_layer_penalty(self.input_layer, m)
                    loss = loss + lam * penalty
                if beta>0:
                    loss = loss + beta*prioritize_penalty(self.prior_inds,m)
                # Add expression penalty if necessary.
                if eta > 0:
                    expressed = torch.mean(torch.sum(x, dim=1))
                    loss = loss + eta * expressed

                # Update mean train loss.
                train_loss = (
                    (N * train_loss + mbsize * loss.item()) / (N + mbsize))
                N += mbsize

                # Gradient step.
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                #scheduler.step()
                self.zero_grad()

                # Adjust temperature.
                self.input_layer.temperature *= r

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.eval()
                val_loss, val_expressed = self.validate(
                    val_loader, loss_fn, lam,beta, eta)
                val_loss, val_expressed = val_loss.item(), val_expressed.item()
                self.train()

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)
                # Update learning rate.
                scheduler.step(val_loss)

                if verbose:
                    print(f'{"-" * 8}Epoch = {epoch + 1}{"-" * 8}')
                    print(f'Train loss = {train_loss:.4f}')
                    print(f'Val loss = {val_loss:.4f}')
                    if eta > 0:
                        print(f'Mean expressed genes = {val_expressed:.4f}')
                    print(input_layer_summary(self.layer_name,self.input_layer,self.L))

            # Update bar.
            if bar:
                tqdm_bar.update(1)

            
            #if not epoch % 20:
                #plt.plot(np.linspace(0, len(self.train_loss), len(self.train_loss)), self.train_loss)
                #plt.title('Training Loss Curve')
                #plt.show()
            

            # Fix input layer if necessary.
            required_fix = input_layer_fix(self.input_layer)

            if not required_fix:
                # Stop early if input layer is converged.
                if input_layer_converged(self.layer_name,self.input_layer,self.L, n_samples=mbsize):
                    if verbose:
                        print('Stopping early: input layer converged')
                    break

                # Stop early if loss converged.
                if loss_early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                    elif (epoch - best_epoch) == lookback:
                        # Skip bar to end.
                        if bar:
                            tqdm_bar.n = max_nepochs

                        if verbose:
                            print('Stopping early: loss converged')
                        break

    def validate(self, loader, loss_fn, lam, beta,eta):
        '''Calculate average loss.'''
        device = next(self.parameters()).device
        mean_loss = 0
        mean_expressed = 0
        N = 0
        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device=device)
                y = y.to(device=device)
                n = len(x)

                # Calculate loss.
                pred, x, m = self.forward(x)
                loss = loss_fn(pred, y)

                # Add penalty term.
                if lam > 0:
                    penalty = input_layer_penalty(self.input_layer, m)
                    loss = loss + lam * penalty
                if beta>0:
                    loss = loss + beta*prioritize_penalty(self.prior_inds,m)
                    
                # Add expression penalty term.
                expressed = torch.mean(torch.sum(x, dim=1))
                if eta > 0:
                    loss = loss + eta * expressed

                mean_loss = (N * mean_loss + n * loss) / (N + n)
                mean_expressed = (N * mean_expressed + n * expressed) / (N + n)
                N += n
            return mean_loss, mean_expressed


def prioritize_penalty(preselected,m):
    
    return torch.mean(torch.sum(torch.tensor(1/2)-torch.min(m[:, preselected], torch.tensor(1/2)),dim=1))




def input_layer_penalty(input_layer, m):
    return torch.mean(torch.sum(m, dim=1))


def input_layer_summary(layer_name,input_layer,L, n_samples=256):
    '''Generate summary string for input layer's convergence.'''
    with torch.no_grad():
        if layer_name == 'binary_gates':
            m = input_layer.sample(n_samples,L)
            mean = torch.mean(m, dim=0)
            dist = torch.min(mean, 1 - mean)
            return 'Mean dist = {:.2f}, Max dist = {:.2f}, Num sel = {}'.format(
                torch.mean(dist).item(),
                torch.max(dist).item(),
                int(torch.sum((mean > 0.5).float()).item()))
        else:
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            sorted_mean = torch.sort(mean, descending=True).values
            relevant = sorted_mean[:input_layer.num_selections]
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                relevant[0].item(), torch.mean(relevant).item(),
                relevant[-1].item())  
        


def input_layer_converged(input_name,input_layer,L, tol=1e-2, n_samples=256):
    '''Determine whether the input layer has converged.'''
    with torch.no_grad():
        if input_name == 'binary_gates':
            m = input_layer.sample(n_samples,L)
            mean = torch.mean(m, dim=0)
            return torch.max(torch.min(mean, 1 - mean)).item() < tol
        else:
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            return (
                torch.sort(mean).values[-input_layer.num_selections].item()
                > 1 - tol)
        
def input_layer_converged_2(input_layer, tol=1e-2, n_samples=256):
    '''Determine whether the input layer has converged.'''
    with torch.no_grad():
        m = input_layer.sample(n_samples)
        mean = torch.mean(m, dim=0)
        return (
                torch.sort(mean).values[-input_layer.num_selections].item()
                > 1 - tol)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def restore_parameters(model, best_model):
    '''Copy parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def input_layer_penalty(input_layer, m):
    if isinstance(input_layer, BinaryGates):
        return torch.mean(torch.sum(m, dim=1))
    else:
        raise ValueError('only BinaryGates layer has penalty')


def input_layer_fix(input_layer):
    '''Fix collisions in the input layer.'''
    required_fix = False

    if isinstance(input_layer, (BinaryMask, ConcreteSelector)):
        # Extract logits.
        logits = input_layer._logits
        argmax = torch.argmax(logits, dim=1).cpu().data.numpy()

        # Locate collisions and reinitialize.
        for i in range(len(argmax) - 1):
            if argmax[i] in argmax[i+1:]:
                required_fix = True
                logits.data[i] = torch.randn(
                    logits[i].shape, dtype=logits.dtype, device=logits.device)
        return required_fix

    return required_fix


def warmstart_model(model, inds):
    '''
    Create model for subset of features by removing parameters.

    Args:
      model: model to copy.
      inds: indices for features to retain.
    '''
    sub_model = deepcopy(model.mlp)
    device = next(model.parameters()).device

    # Resize input layer.
    layer = sub_model.fc[0]
    new_layer = nn.Linear(len(inds), layer.out_features).to(device)
    new_layer.weight.data = layer.weight[:, inds]
    new_layer.bias.data = layer.bias
    sub_model.fc[0] = new_layer

    return sub_model


class MLP_2(nn.Module):


    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU()):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation function.
        self.activation = activation

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        return self.fc[-1](x)

    def fit(self,
            train_dataset,
            val_dataset,
            mbsize,
            max_nepochs,
            loss_fn,
            lr=1e-3,
            min_lr=1e-5,
            lr_factor=0.5,
            optimizer='Adam',
            lookback=10,
            bar=False,
            verbose=True):

        # Set up optimizer.
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)

        # Set up data loaders.
        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(
            train_dataset, batch_size=mbsize, shuffle=True, drop_last=True,
            worker_init_fn=train_init, num_workers=4)
        val_loader = DataLoader(
            val_dataset, batch_size=mbsize, worker_init_fn=val_init,
            num_workers=4)

        # Determine device.
        device = next(self.parameters()).device

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []
        best_model = None
        best_loss = np.inf
        best_epoch = None

        # Bar setup.
        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=True)

        # Begin training.
        for epoch in range(max_nepochs):
            # For tracking mean train loss.
            train_loss = 0
            N = 0

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Forward pass.
                pred = self.forward(x)

                # Calculate loss.
                loss = loss_fn(pred, y)

                # Update mean train loss.
                train_loss = (
                    (N * train_loss + mbsize * loss.item()) / (N + mbsize))
                N += mbsize

                # Gradient step.
                loss.backward()
                optimizer.step()
                self.zero_grad()

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.eval()
                val_loss = self.validate(val_loader, loss_fn).item()
                self.train()

                # Update learning rate.
                scheduler.step(val_loss)

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)

                if verbose:
                    print(f'{"-" * 8}Epoch = {epoch + 1}{"-" * 8}')
                    print(f'Train loss = {train_loss:.4f}')
                    print(f'Val loss = {val_loss:.4f}')

            # Update bar.
            if bar:
                tqdm_bar.update(1)

            # Check for early stopping.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self)
                best_epoch = epoch
            elif (epoch - best_epoch) == lookback:
                # Skip bar to end.
                if bar:
                    tqdm_bar.n = max_nepochs

                if verbose:
                    print('Stopping early')
                break

        # Restore model parameters.
        restore_parameters(self, best_model)

    def validate(self, loader, loss_fn):
        '''Calculate average loss.'''
        device = next(self.parameters()).device
        mean_loss = 0
        N = 0

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device=device)
                y = y.to(device=device)
                n = len(x)

                # Calculate loss.
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                mean_loss = (N * mean_loss + n * loss) / (N + n)
                N += n

        return mean_loss


class SelectorMLP_2(nn.Module):
    '''
    MLP model with embedded selector layer.

    Args:
      input_layer: selection layer type (e.g., BinaryMask).
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between layers.
      preselected_inds: feature indices that are already selected.
      num_selections: number of features to select (for BinaryMask and
        ConcreteSelector layers).
      kwargs: additional arguments for input layers.
    '''

    def __init__(self,
                 input_layer,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU(),
                 preselected_inds=[],
                 num_selections=None,
                 **kwargs):
        # Verify arguments.
        super().__init__()
        if num_selections is None:
            if input_layer in ('binary_mask', 'concrete_selector'):
                raise ValueError(
                    f'must specify num_selections for {input_layer} layer')
        else:
            if input_layer in ('binary_gates'):
                raise ValueError('num_selections cannot be specified for '
                                 f'{input_layer} layer')

        # Set up for pre-selected features.
        preselected_inds = np.sort(preselected_inds)
        assert len(preselected_inds) < input_size
        self.preselected = np.array(
            [i in preselected_inds for i in range(input_size)])
        preselected_size = len(preselected_inds)
        self.has_preselected = preselected_size > 0

        # Set up input layer.
        if input_layer == 'binary_mask':
            mlp_input_size = input_size
            self.input_layer = BinaryMask(
                input_size - preselected_size, num_selections, **kwargs)
        elif input_layer == 'binary_gates':
            mlp_input_size = input_size
            self.input_layer = BinaryGates(
                input_size - preselected_size, **kwargs)
        elif input_layer == 'concrete_selector':
            mlp_input_size = num_selections + preselected_size
            self.input_layer = layers.ConcreteSelector(
                input_size - preselected_size, num_selections, **kwargs)
        else:
            raise ValueError('unsupported input layer: {}'.format(input_layer))

        # Create MLP.
        self.mlp = MLP_2(mlp_input_size, output_size, hidden, activation)

    def forward(self, x):
        '''Apply input layer and return MLP output.'''
        if self.has_preselected:
            pre = x[:, self.preselected]
            x, m = self.input_layer(x[:, ~self.preselected])
            x = torch.cat([pre, x], dim=1)
        else:
            x, m = self.input_layer(x)
        pred = self.mlp(x)
        return pred, x, m

    def fit(self,
            train_dataset,
            val_dataset,
            lr,
            mbsize,
            max_nepochs,
            start_temperature,
            end_temperature,
            loss_fn,
            eta=0,
            lam=0,
            optimizer='Adam',
            lookback=10,
            bar=False,
            verbose=True):
        '''
        Train the model.

        Args:
          train_dataset: training dataset.
          val_dataset: validation dataset.
          lr: learning rate.
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          start_temperature:
          end_temperature:
          loss_fn: loss function.
          eta: penalty parameter for number of expressed genes.
          lam: penalty parameter.
          optimizer: optimizer type.
          lookback: number of epochs to wait for improvement before stopping.
          bar: whether to display tqdm progress bar.
          verbose: verbosity.
        '''
        # Verify arguments.
        if lam != 0:
            if not isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam should only be specified when using '
                                 'BinaryGates layer')
        else:
            if isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam must be specified when using '
                                 'BinaryGates layer')
        if eta > 0:
            if isinstance(self.input_layer, BinaryGates):
                raise ValueError('lam cannot be specified when using '
                                 'BinaryGates layer')

        if end_temperature > start_temperature:
            raise ValueError('temperature should be annealed downwards, must '
                             'have end_temperature <= start_temperature')
        elif end_temperature == start_temperature:
            loss_early_stopping = True
        else:
            loss_early_stopping = False

        # Set up optimizer.
        optimizer = optimizer = optim.Adam(self.parameters(), lr=lr)

        # Set up data loaders.
        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(train_dataset, batch_size=mbsize,
                                  shuffle=True, drop_last=True,
                                  worker_init_fn=train_init, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=mbsize,
                                worker_init_fn=val_init, num_workers=4)

        # Determine device.
        device = next(self.parameters()).device

        # Set temperature and determine rate for decreasing.
        self.input_layer.temperature = start_temperature
        r = np.power(end_temperature / start_temperature,
                     1 / ((len(train_dataset) // mbsize) * max_nepochs))

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []
        best_loss = np.inf
        best_epoch = -1

        # Bar setup.
        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=False)

        # Begin training.
        for epoch in range(max_nepochs):
            # For tracking mean train loss.
            train_loss = 0
            N = 0

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Calculate loss.
                pred, x, m = self.forward(x)
                loss = loss_fn(pred, y)

                # Calculate penalty if necessary.
                if lam > 0:
                    penalty = input_layer_penalty(self.input_layer, m)
                    loss = loss + lam * penalty
                    
                # Add expression penalty if necessary.
                if eta > 0:
                    expressed = torch.mean(torch.sum(x, dim=1))
                    loss = loss + eta * expressed

                # Update mean train loss.
                train_loss = (
                    (N * train_loss + mbsize * loss.item()) / (N + mbsize))
                N += mbsize

                # Gradient step.
                loss.backward()
                optimizer.step()
                self.zero_grad()

                # Adjust temperature.
                self.input_layer.temperature *= r

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.eval()
                val_loss, val_expressed = self.validate(
                    val_loader, loss_fn, lam, eta)
                val_loss, val_expressed = val_loss.item(), val_expressed.item()
                self.train()

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)


            # Update bar.
            if bar:
                tqdm_bar.update(1)

            # Fix input layer if necessary.
            required_fix = input_layer_fix(self.input_layer)

            if not required_fix:
                # Stop early if input layer is converged.
                if input_layer_converged_2(self.input_layer, n_samples=mbsize):
                    if verbose:
                        print('Stopping early: input layer converged')
                    break

                # Stop early if loss converged.
                if loss_early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                    elif (epoch - best_epoch) == lookback:
                        # Skip bar to end.
                        if bar:
                            tqdm_bar.n = max_nepochs

                        if verbose:
                            print('Stopping early: loss converged')
                        break

    def validate(self, loader, loss_fn, lam, eta):
        '''Calculate average loss.'''
        device = next(self.parameters()).device
        mean_loss = 0
        mean_expressed = 0
        N = 0
        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device=device)
                y = y.to(device=device)
                n = len(x)

                # Calculate loss.
                pred, x, m = self.forward(x)
                loss = loss_fn(pred, y)

                # Add penalty term.
                if lam > 0:
                    penalty = input_layer_penalty(self.input_layer, m)
                    loss = loss + lam * penalty
                    
                # Add expression penalty term.
                expressed = torch.mean(torch.sum(x, dim=1))
                if eta > 0:
                    loss = loss + eta * expressed

                mean_loss = (N * mean_loss + n * loss) / (N + n)
                mean_expressed = (N * mean_expressed + n * expressed) / (N + n)
                N += n

        return mean_loss, mean_expressed
