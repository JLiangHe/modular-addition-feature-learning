from utils import *
from collections import deque
import plotly.graph_objects as go

# Read Configurations from configs.yaml
configs = read_config()
pipeline_config = Config(configs)
print(pipeline_config.__dict__)

world = Trainer(config=pipeline_config)
print(f'Run name {world.run_name}')
world.initial_save_if_appropriate()

recent_test_loss = deque(maxlen=2)
save_point = 0

for epoch in range(pipeline_config.num_epochs):
    # Perform a training step and get train/test losses
    train_loss, test_loss = world.do_a_training_step(epoch)
        
    # Stop training if test loss falls below the threshold
    if test_loss.item() < pipeline_config.stopping_thresh:
        break
        
    # Save model state if it's time to do so
    if pipeline_config.is_it_time_to_save(epoch=epoch):
        world.save_epoch(epoch=epoch)
        if epoch % 1000 == 0:
            world.save_epoch(epoch=epoch, local_save=True)
        # Dynamically update weight decay

# Save final model state after training is complete
world.post_training_save(save_optimizer_and_scheduler=True)