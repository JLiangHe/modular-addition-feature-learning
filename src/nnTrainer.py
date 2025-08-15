import yaml, os, time, wandb
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from typing import Optional

from model_base import EmbedMLP
from utils import Config, gen_train_test, full_loss, acc, cross_entropy_high_precision


class Trainer:
    '''Trainer class for managing the training process of a model'''

    def __init__(self, config: Config, model: Optional[EmbedMLP] = None) -> None:
               
        # Use a given model or initialize a new Transformer model with the provided config
        self.model = model if model is not None else EmbedMLP(
                        d_vocab=config.d_vocab,
                        d_model=config.d_model,
                        d_mlp=config.d_mlp,
                        act_type=config.act_type,
                        use_cache=False,
                        init_type=config.init_type,
                        init_scale=config.init_scale if hasattr(config, 'init_scale') else 0.1,
                        embed_type=config.embed_type
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
        init_scale_str = f"scale_{config.init_scale}" if hasattr(config, 'init_scale') else ""
        self.run_name = f"p_{config.p}_dmlp_{config.d_mlp}_{config.act_type}_{config.init_type}_{init_scale_str}_decay_{config.weight_decay}_{formatted_time}"
        
        # Initialize experiment logging with wandb (Weights and Biases)
        wandb.init(project="modular_addition", config=config, name=self.run_name)
        
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
