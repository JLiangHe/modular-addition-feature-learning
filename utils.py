import yaml, os, time, wandb, random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

from model_base import *
#from model_base_simple import *


########## Configuration Managers ##########
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
        
        # Load configurations from the dictionary and set them as attributes
        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

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
    random.seed(config.seed)
    random.shuffle(pairs)
    
    # If frac_train is 1, use the whole dataset for both train and test.
    if config.frac_train == 1:
        return pairs, pairs
    
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]

########## Training Managers ##########
class Trainer:
    '''Trainer class for managing the training process of a model'''

    def __init__(self, config: Config, model: Optional[EmbedMLP] = None) -> None:
               
        # Use a given model or initialize a new Transformer model with the provided config
        self.model = model if model is not None else EmbedMLP(
                        d_vocab=config.d_vocab,
                        d_model=config.d_model,
                        d_mlp=config.d_mlp,
                        act_type=config.act_type,
                        use_cache=False
                    )
        self.model.to(config.device)  # Move model to specified device (e.g., GPU)
        if config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.98)
            )

            # Update scheduler with `AdamW` optimizer
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(step / 10, 1))
        elif config.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay  # This applies L2 regularization, equivalent to weight decay in GD
            )

            # You can keep the scheduler as is, if desired
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(step / 10, 1))
        
        # Generate a unique run name for this training session
        formatted_time = time.strftime("%m%d%H%M", time.localtime())
        self.run_name = f"p_{config.p}_dmodel_{config.d_model}_dmlp_{config.d_mlp}_act_{config.act_type}_decay_{config.weight_decay}_fractrain_{config.frac_train}_DFT_{formatted_time}"
        
        # Initialize experiment logging with wandb (Weights and Biases)
        wandb.init(project="modular_addition_feature_learning", config=config, name=self.run_name)
        
        # Define the directory where model checkpoints will be saved
        self.save_dir = "saved_models"
        os.makedirs(os.path.join(self.save_dir, self.run_name), exist_ok=True)

        # Generate training and testing datasets
        self.train, self.test = gen_train_test(config=config)

        # Save the training and testing datasets
        train_path = os.path.join(self.save_dir, self.run_name, "train_data.pth")
        test_path = os.path.join(self.save_dir, self.run_name, "test_data.pth")
        torch.save(self.train, train_path)
        torch.save(self.test, test_path)
        
        # Dictionary to store metrics (train/test losses, etc.)
        self.metrics_dictionary = defaultdict(dict)
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        
        # Lists to store loss values during training
        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []
        self.param_norms = []
        self.test_accs = []
        self.train_accs = []
        self.config = config

    def save_epoch(self, epoch, save_to_wandb=True, local_save=False):
        '''Save model and training state at the specified epoch'''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'grad_norm': self.grad_norms[-1],
            'param_norm': self.param_norms[-1],
            'test_accuracy': self.test_accs[-1],
            'train_accuracy': self.train_accs[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)  # Log to wandb
            config_dict = {
                k: (str(v) if isinstance(v, torch.device) else v)
                for k, v in self.config.__dict__.items()
            }
            wandb.log(config_dict) 
            print("Saved epoch to wandb")
        if self.config.save_models or local_save: 
            # Save model state to a file
            save_path = os.path.join(self.save_dir, self.run_name, f"{epoch}.pth")
            torch.save(save_dict, save_path)
            print(f"Saved model to {save_path}")
        self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''Perform a single training step and return train and test loss'''
        # Calculate training loss on the training data
        train_loss = full_loss(config=self.config, model=self.model, data=self.train)
        
        # Calculate testing loss on the testing data
        test_loss = full_loss(config=self.config, model=self.model, data=self.test)

        # Calculate training loss on the training data
        train_acc = acc(config=self.config, model=self.model, data=self.train)
        
        # Calculate testing loss on the testing data
        test_acc = acc(config=self.config, model=self.model, data=self.test)

        # Append loss values to tracking lists
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        
        if epoch % 100 == 0:
            # Log progress every 100 epochs
            print(f'Epoch {epoch}, train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}')

        
        # Backpropagation and optimization step
        train_loss.backward()  # Compute gradients
        # Compute gradient norm and parameter norm
        grad_norm = 0.0
        param_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item()**2  # Sum of squared gradients
            param_norm += param.norm(2).item()**2  # Sum of squared parameters
        self.grad_norms.append(grad_norm**0.5)  # L2 norm of gradients
        self.param_norms.append(param_norm**0.5)  # L2 norm of parameters

        self.optimizer.step()  # Update model parameters
        self.scheduler.step()  # Update learning rate
        self.optimizer.zero_grad()  # Clear gradients
        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        '''Save initial model state and data if configured to do so'''
        if self.config.save_models:
            save_path = os.path.join(self.save_dir, self.run_name, 'init.pth')
            save_dict = {
                'model': self.model.state_dict(),
                'train_data': self.train,
                'test_data': self.test
            }
            torch.save(save_dict, save_path)

    def post_training_save(self, save_optimizer_and_scheduler=True, log_to_wandb=True):
        '''Save final model state and metrics after training'''
        save_path = os.path.join(self.save_dir, self.run_name, "final.pth")
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'grad_norms': self.grad_norms,
            'param_norms': self.param_norms,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            # Optionally save optimizer and scheduler states
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        torch.save(save_dict, save_path)
        print(f"Saved model to {save_path}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)

    def take_metrics(self, train, epoch):
        '''Calculate and log metrics for the current epoch'''
        with torch.inference_mode():  # Disable gradient calculation for metrics
            def sum_sq_weights():
                '''Calculate sum of squared weights (example metric)'''
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            #print('taking metrics')

            # Prepare all possible data points for metric calculations
            all_data = torch.tensor([(i, j, self.config.p) for i in range(self.config.p) for j in range(self.config.p)]).to(self.config.device)
            
            # Calculate frequency-related metrics (customized calculations)
            key_freqs = calculate_key_freqs(config=self.config, model=self.model, all_data=all_data)
            logits = self.model(all_data)[:, -1, :-1]
            fourier_basis = make_fourier_basis(config=self.config)
            is_train, is_test = self.config.is_train_is_test(train=train)
            labels = torch.tensor([self.config.fn(i, j) for i, j, _ in all_data]).to(self.config.device)

            # Compile metrics to log
            metrics = {
                'epoch': epoch, 
                'trig_loss': calculate_trig_loss(
                    config=self.config,
                    model=self.model,
                    train=train,
                    key_freqs=key_freqs,
                    is_test=is_test,
                    is_train=is_train,
                    labels=labels,
                    logits=logits,
                    fourier_basis=fourier_basis,
                    all_data=all_data
                ),
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': calculate_excluded_loss(
                    logits=logits,
                    key_freqs=key_freqs,
                    fourier_basis=fourier_basis,
                    is_train=is_train,
                    config=self.config,
                    is_test=is_test,
                    labels=labels
                ),
                'coefficients': calculate_coefficients(
                    p=self.config.p, 
                    logits=logits, 
                    fourier_basis=fourier_basis, 
                    key_freqs=key_freqs, 
                    device=self.config.device
                ),
            }
            wandb.log(metrics)  # Log metrics to wandb
            self.metrics_dictionary[epoch].update(metrics)

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
    logits = model(data)
    labels = torch.tensor([config.fn(i, j) for i, j in data]).to(config.device)
    predictions = torch.argmax(logits, dim=1)  # Get predicted class indices
    correct = (predictions == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Calculate accuracy
    return accuracy

