import yaml, os, time, wandb, random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

try:
    # For running from within src/ directory (script execution)
    from mechanism_base import *
    from model_base import EmbedMLP
except ImportError:
    # For running from parent directory (Jupyter notebooks)
    from src.mechanism_base import *
    from src.model_base import EmbedMLP


########## Configuration Managers ##########
def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy random
    torch.manual_seed(seed)                    # PyTorch CPU random
    torch.cuda.manual_seed(seed)               # PyTorch GPU random (current GPU)
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU random (all GPUs)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable CuDNN benchmarking for reproducibility
    
def read_config():
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "configs.yaml")
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

@dataclass
class Config:
    def __init__(self, config):
        # Ensure that the config dictionary is provided
        if not config:
            raise ValueError("Configuration dictionary cannot be None or empty.")
        
        # Load configurations from the nested dictionary structure and flatten them
        self._flatten_config(config)
        
        # Ensure numeric types are properly converted
        if hasattr(self, 'lr') and isinstance(self.lr, str):
            self.lr = float(self.lr)
        if hasattr(self, 'weight_decay') and isinstance(self.weight_decay, str):
            self.weight_decay = float(self.weight_decay)
        if hasattr(self, 'stopping_thresh') and isinstance(self.stopping_thresh, str):
            self.stopping_thresh = float(self.stopping_thresh)

        # Set d_vocab equal to p if not explicitly specified
        if not hasattr(self, 'd_vocab') or self.d_vocab is None:
            self.d_vocab = self.p
        
        # Set d_model for one_hot embedding type
        if not hasattr(self, 'd_model') or self.d_model is None:
            if hasattr(self, 'embed_type') and self.embed_type == 'one_hot':
                self.d_model = self.d_vocab
            else:
                # For learned embeddings, use a default dimension
                self.d_model = 128
        
        # Set all random seeds for reproducibility
        if hasattr(self, 'seed'):
            set_all_seeds(self.seed)
            print(f"All random seeds set to: {self.seed}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def _flatten_config(self, config_dict, parent_key=''):
        """Flatten nested configuration dictionary into flat attributes"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                self._flatten_config(value, parent_key)
            else:
                # Set the attribute directly
                setattr(self, key, value)

    # Property to generate a matrix of random answers (used for 'rand' function)
    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    # Property to map function names to their corresponding mathematical operations
    @property 
    def fns_dict(self):
        return {
            'add': lambda x, y: (x + y) % self.p, # Addition modulo p
            'subtract': lambda x, y: (x - y) % self.p, # Subtraction modulo p
            'x2xyy2': lambda x, y: (x**2 + x * y + y**2) % self.p, # Polynomial function modulo p
            'rand': lambda x, y: self.random_answers[x][y] # Random value from a precomputed table
        }

    # Property to access the selected function based on 'fn_name'
    @property
    def fn(self):
        return self.fns_dict[self.fn_name]

    # Function to create Boolean arrays indicating if a data point is in the training or test set
    def is_train_is_test(self, train):
        '''Creates an array of Boolean indices according to whether each data point is in train or test.
        Used to index into the big batch of all possible data'''
        # Initialize empty lists for training and test indices
        is_train = []
        is_test = []
        # Iterate over all possible data points (0 <= x, y < p)
        for x in range(self.p):
            for y in range(self.p):
                if (x, y, 113) in train: # If the data point is in the training set
                    is_train.append(True)
                    is_test.append(False)
                else: # Otherwise, it's in the test set
                    is_train.append(False)
                    is_test.append(True)
        # Convert lists to NumPy arrays for efficient indexing
        is_train = np.array(is_train)
        is_test = np.array(is_test)
        return (is_train, is_test)

    # Function to determine if it's time to save the model (based on epoch number)
    def is_it_time_to_save(self, epoch):
        return (epoch % self.save_every == 0)

    # Function to determine if it's time to take metrics (based on epoch number)
    def is_it_time_to_take_metrics(self, epoch):
        return epoch % self.take_metrics_every_n_epochs == 0

    def update_param(self, param_name, value):
        setattr(self, param_name, value)

########## Data Manager ##########
def gen_train_test(config: Config):
    '''Generate train and test split'''
    num_to_generate = config.p
    pairs = [(i, j) for i in range(num_to_generate) for j in range(num_to_generate)]
    # Note: Global seeding already handled in Config.__init__, but keeping for extra safety
    random.seed(config.seed)
    random.shuffle(pairs)
    
    # If frac_train is 1, use the whole dataset for both train and test.
    if config.frac_train == 1:
        return pairs, pairs
    
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]

########## Training Managers ##########
# Trainer class has been moved to NNTrainer.py

########## Loss Definition ##########
def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def full_loss(config : Config, model: EmbedMLP, data):
    '''Takes the cross entropy loss of the model on the data'''
    # Convert data to tensor and move to correct device
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, device=config.device)
    elif data.device != config.device:
        data = data.to(config.device)
    
    # Take the final position only
    logits = model(data)#[:, -1]
    labels = torch.tensor([config.fn(i, j) for i, j in data]).to(config.device)
    return cross_entropy_high_precision(logits, labels)

def acc_rate(logits, labels):
    predictions = torch.argmax(logits, dim=1)  # Get predicted class indices
    correct = (predictions == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Calculate accuracy
    return accuracy

def acc(config: Config, model: EmbedMLP, data):
    # Convert data to tensor and move to correct device
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, device=config.device)
    elif data.device != config.device:
        data = data.to(config.device)
    
    logits = model(data)
    labels = torch.tensor([config.fn(i, j) for i, j in data]).to(config.device)
    predictions = torch.argmax(logits, dim=1)  # Get predicted class indices
    correct = (predictions == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Calculate accuracy
    return accuracy

