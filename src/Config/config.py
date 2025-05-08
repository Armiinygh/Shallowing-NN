from types import SimpleNamespace
from torch import nn 

cfg = SimpleNamespace(**{})

cfg.epochs = 20
cfg.batch_size = 128
cfg.learning_rate = 0.00001
cfg.output_size = 256



cfg.loss_function = nn.CrossEntropyLoss() #[nn.CrossEntropyLoss(),  nn.NLLLoss(), ]
cfg.activation_function = nn.ReLU() #[nn.LeakyReLU() , nn.GELU(), nn.ELU()]
# run config
cfg.save_model = True

"""
Configuration file for model training.

Stores training parameters and NN hyperparameters in a SimpleNamespace object (cfg),
accessible throughout the project.
"""
from types import SimpleNamespace
from torch import nn 

cfg = SimpleNamespace(**{})

cfg.epochs = 20                # Number of training epochs
cfg.batch_size = 128           # Samples per batch
cfg.learning_rate = 0.00001    # Learning rate for optimizer
cfg.output_size = 256          # Hidden layer size for shallow net

cfg.loss_function = nn.CrossEntropyLoss() #[nn.CrossEntropyLoss(),  nn.NLLLoss(), ] 
cfg.activation_function = nn.ReLU() #[nn.LeakyReLU() , nn.GELU(), nn.ELU()]

cfg.save_model = True          # Save model after training