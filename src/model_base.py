import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch, einops

########## Hook Manager ##########
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        # Lists to store forward and backward hook handles for later removal
        self.fwd_hooks = []  # Forward hooks
        self.bwd_hooks = []  # Backward hooks
    
    def give_name(self, name):
        # Sets a name for the hook point (called during model initialization for tracking)
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Adds a hook to the module (either forward or backward)
        # hook is a function that takes (activation, hook_name) as input
        # Converts it to PyTorch's hook format, which takes module, input, and output
        def full_hook(module, module_input, module_output):
            # Calls the provided hook function, passing the output (activation) and hook name
            return hook(module_output, name=self.name)
        
        if dir == 'fwd':
            # Registers the full hook as a forward hook and stores the handle
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == 'bwd':
            # Registers the full hook as a backward hook and stores the handle
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")  # Raise error if direction is invalid
    
    def remove_hooks(self, dir='fwd'):
        # Removes all hooks of the specified direction
        if (dir == 'fwd') or (dir == 'both'):
            for hook in self.fwd_hooks:
                hook.remove()  # Remove each forward hook
            self.fwd_hooks = []
        if (dir == 'bwd') or (dir == 'both'):
            for hook in self.bwd_hooks:
                hook.remove()  # Remove each backward hook
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")  # Raise error if direction is invalid
    
    def forward(self, x):
        # By default, acts as an identity function, simply returning its input
        return x

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model, embed_type='one_hot'):
        super().__init__()
        self.d_vocab = d_vocab
        self.embed_type = embed_type
        
        if embed_type == 'learned':
            # Weight matrix for embedding, initialized with standard deviation scaled by sqrt(d_model)
            self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
        elif embed_type == 'one_hot':
            # For one-hot embeddings, we don't need learnable parameters
            self.W_E = None
        else:
            raise ValueError(f"Invalid embed_type: {embed_type}. Must be 'one_hot' or 'learned'")
    
    def forward(self, x):
        # Convert input tokens to embedded vectors
        # Input x is expected to be of shape (batch_size, 2), indexing tokens in the vocabulary
        # Convert input to a tensor if it's not already
        if isinstance(x, list):
            device = self.W_E.device if self.W_E is not None else 'cpu'
            x = torch.tensor(x, device=device)
        # Validate shape
        assert x.ndim == 2 and x.shape[1] == 2, f"Expected input shape (batch_size, 2), got {x.shape}"

        if self.embed_type == 'one_hot':
            # One-hot embedding: sum the one-hot vectors for the two input tokens
            embed = F.one_hot(x, num_classes=self.d_vocab).float().sum(dim=1).unsqueeze(1)
        elif self.embed_type == 'learned':
            # Learned embedding: use embedding matrix to get vectors and sum them
            embed = torch.einsum('dbp -> bpd', self.W_E[:, x]).sum(dim=1).unsqueeze(1)
        
        return embed

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        # Learnable scale and shift parameters
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            # Normalize the input
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            # Apply learnable scale and shift
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, d_vocab, act_type, model, init_type='random', init_scale=0.1):
        super().__init__()
        self.model = model
        self.init_type = init_type
        self.init_scale = init_scale
        
        # Initialize weights based on init_type
        if init_type == 'random':
            # Random initialization
            self.W_in = nn.Parameter(self.init_scale * torch.randn(d_mlp, d_model)/np.sqrt(d_model))
            self.W_out = nn.Parameter(self.init_scale * torch.randn(d_vocab, d_mlp)/np.sqrt(d_model))
        elif init_type == 'single-freq':
            # Sparse frequency-based initialization
            freq_num = (d_vocab-1)//2
            init_freq = decide_frequencies(d_mlp, d_model, freq_num)
            fourier_basis, _ = get_fourier_basis(d_vocab)
            
            self.W_in = nn.Parameter(self.init_scale * np.sqrt(d_vocab/2) * sparse_initialization(d_mlp, d_model, init_freq) @ fourier_basis)
            self.W_out = nn.Parameter(self.init_scale * np.sqrt(d_vocab/2) * fourier_basis.T @ sparse_initialization(d_mlp, d_model, init_freq).T)
        else:
            raise ValueError(f"Invalid init_type:  ini{init_type}. Must be 'random' or 'single-freq'")
        
        # Store activation - can be string or function
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        
        # Check if act_type is a string or a callable function
        if isinstance(act_type, str):
            assert act_type in ['ReLU', 'GeLU', 'Quad', 'Id'], f"Invalid activation type: {act_type}"
        elif not callable(act_type):
            raise ValueError("act_type must be either a string ('ReLU', 'GeLU', 'Quad', 'Id') or a callable function")
        
        fourier_basis, _ = get_fourier_basis(d_vocab)
        self.register_buffer('basis', fourier_basis.clone().detach())
    
    def forward(self, x):
        # Linear transformation and activation
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x))
        
        # Apply activation function - either built-in or custom
        if callable(self.act_type):
            # Custom activation function
            x = self.act_type(x)
        elif self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        elif self.act_type == "Quad":
            x = torch.square(x)
        elif self.act_type == "Id":
            x = x
            
        x = self.hook_post(x)
        # Output transformation
        x = torch.einsum('dm,bpm->bpd', self.W_out, x)
        return x

class EmbedMLP(nn.Module):
    def __init__(self, d_vocab, d_model, d_mlp, act_type, use_cache=False, use_ln=True, init_type='random', init_scale=0.1, embed_type='one_hot'):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.init_type = init_type

        # Embedding layers
        self.embed = Embed(d_vocab, d_model, embed_type=embed_type)
        self.mlp = MLP(d_model, d_mlp, d_vocab, act_type, model=[self], init_type=init_type, init_scale=init_scale)
        # Optional layer normalization at the output
        # self.ln = LayerNorm(d_model, model=[self])
        # Unembedding layer for output logits
        # self.unembed = Unembed(self.embed)#Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        # Assign names to hook points for easier debugging and monitoring
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        # Pass input through embedding layers
        x = self.embed(x)
        # Pass input through MLP
        x = self.mlp(x)  
        # Optional normalization (commented out)
        # x = self.ln(x)
        # Pass through unembedding layer
        # x = self.unembed(x)
        return x.squeeze(1)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        # Gather all hook points in the model for easy access
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        # Remove all hooks for cleaner training or evaluation
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name + '_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')


########## Auxiliary Functions ##########
def get_fourier_basis(p):
    # Initialize the list to store Fourier basis vectors and names
    fourier_basis = []
    fourier_basis_names = []

    # Add the constant term (normalized)
    fourier_basis.append(torch.ones(p) / np.sqrt(p))
    fourier_basis_names.append('Const')

    # Generate Fourier basis for cosines and sines
    for i in range(1, p // 2 + 1):
        # Compute cosine and sine basis terms
        cosine = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sine = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        # Normalize each basis function
        cosine /= cosine.norm()
        sine /= sine.norm()
        # Append basis vectors and their names
        fourier_basis.append(cosine)
        fourier_basis.append(sine)
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    
    # Special case for even p: cos(k*pi), alternating +1 and -1
    if p % 2 == 0:
        cosine = torch.cos(torch.pi * torch.arange(p))
        cosine /= cosine.norm()
        fourier_basis.append(cosine)
        fourier_basis_names.append(f'cos {p // 2}')
    
    # Stack the basis vectors into a matrix and move to the desired device
    fourier_basis = torch.stack(fourier_basis, dim=0)
    
    return fourier_basis, fourier_basis_names

def decide_frequencies(d_mlp, d_model, freq_num):
    """
    Decide frequency assignments for each neuron.
    
    For a weight matrix of shape (d_mlp, d_model), valid frequencies are integers 
    in the range [1, (d_model-1)//2]. This function samples 'freq_num' unique frequencies 
    uniformly from this range and assigns them to the neurons as equally as possible.
    
    Args:
        d_mlp (int): Number of neurons (rows).
        d_model (int): Number of columns in the weight matrix.
        freq_num (int): Number of unique frequencies to sample.
    
    Returns:
        np.ndarray: A 1D array of length d_mlp containing the frequency assigned to each neuron.
    """
    # Determine the maximum available frequency.
    max_freq = (d_model - 1) // 2
    if freq_num > max_freq:
        raise ValueError(f"freq_num ({freq_num}) cannot exceed the number of available frequencies ({max_freq}).")
    
    # Sample 'freq_num' unique frequencies uniformly from 1 to max_freq.
    freq_choices = np.random.choice(np.arange(1, max_freq + 1), size=freq_num, replace=False)
    
    # Assign neurons equally among the chosen frequencies.
    # Repeat the frequency choices until we have at least d_mlp assignments.
    repeats = (d_mlp + freq_num - 1) // freq_num  # Ceiling division.
    freq_assignments = np.tile(freq_choices, repeats)[:d_mlp]
    
    # Shuffle to randomize the order of assignments.
    np.random.shuffle(freq_assignments)
    
    return freq_assignments

def sparse_initialization(d_mlp, d_model, freq_assignments):
    """
    Generate a sparse weight matrix using the provided frequency assignments.
    
    For each neuron (row) assigned frequency f, this function assigns Gaussian random values 
    to columns (2*f - 1) and (2*f) of that row. All other entries remain zero.
    
    Args:
        d_mlp (int): Number of neurons (rows) in the weight matrix.
        d_model (int): Number of columns in the weight matrix.
        freq_assignments (np.ndarray): 1D array of length d_mlp containing the frequency for each neuron.
    
    Returns:
        torch.Tensor: A weight matrix of shape (d_mlp, d_model) with the sparse initialization.
    """
    # Create a weight matrix filled with zeros.
    weight = torch.zeros(d_mlp, d_model)
    
    # For each neuron, assign Gaussian random values to the corresponding columns.
    for i, f in enumerate(freq_assignments):
        col1 = 2 * f - 1
        col2 = 2 * f
        # Check that the computed columns are within bounds.
        if col2 < d_model:
            vec = torch.randn(2, device=weight.device, dtype=weight.dtype)
            # Normalize to have L2 norm = 1
            vec = vec / torch.norm(vec, p=2)
    
            # Assign the two normalized components
            weight[i, col1] = vec[0]
            weight[i, col2] = vec[1]
        else:
            # This branch should not be reached if f is chosen correctly.
            raise IndexError(f"Computed column index {col2} is out of bounds for d_model={d_model}.")
    
    return weight