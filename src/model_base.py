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
    def __init__(self, d_vocab, d_model):
        super().__init__()
        # Weight matrix for embedding, initialized with standard deviation scaled by sqrt(d_model)
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
        self.d_vocab = d_vocab
    
    def forward(self, x):
        # Convert input tokens to embedded vectors using a matrix product
        # Input x is expected to be of shape (batch_size, 2), indexing tokens in the vocabulary
        # Convert input to a tensor if it's not already
        if isinstance(x, list):
            x = torch.tensor(x, device=self.W_E.device)
        # Validate shape
        assert x.ndim == 2 and x.shape[1] == 2, f"Expected input shape (batch_size, 2), got {x.shape}"

        embed = F.one_hot(x, num_classes=self.d_vocab).float().sum(dim=1).unsqueeze(1)
        #embed = torch.einsum('dbp -> bpd', self.W_E[:, x]).sum(dim=1).unsqueeze(1)  # This operation gathers the embeddings for the input tokens      
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
    def __init__(self, d_model, d_mlp, d_vocab, act_type, model):
        super().__init__()
        self.model = model
        # Input and output weight matrices for the feedforward layer
        self.W_in = nn.Parameter(0.01 * torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        #self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(0.01 * torch.randn(d_vocab, d_mlp)/np.sqrt(d_model))
        #self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU', 'Quad', 'Id']
        fourier_basis, _ = get_fourier_basis(d_vocab)
        self.register_buffer('basis', torch.tensor(fourier_basis))
    
    def quad_act(self, x):
        return x**2

    def forward(self, x):
        # Linear transformation and activation
        #x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in @ self.basis, x))
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x))
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        elif self.act_type == "Quad":
            x = self.quad_act(x)
        elif self.act_type == "Id":
            x = x
        x = self.hook_post(x)
        # Output transformation
        #x = torch.einsum('dm,bpm->bpd', self.basis.T @ self.W_out, x) #+ self.b_out
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) #+ self.b_out
        return x

class EmbedMLP(nn.Module):
    def __init__(self, d_vocab, d_model, d_mlp, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        # Embedding layers
        self.embed = Embed(d_vocab, d_model)
        self.mlp = MLP(d_model, d_mlp, d_vocab, act_type, model=[self])
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
