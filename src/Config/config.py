from torch import nn
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.epochs = 5
cfg.batch_size = 128
cfg.learning_rate = 0.00001
cfg.output_size = 256


cfg.optimization = "Adam"  # ["Adam" , "SGD"]


cfg.loss_function = nn.CrossEntropyLoss()  # [nn.CrossEntropyLoss(),  nn.NLLLoss(), ]
cfg.activation_function = nn.ReLU()  # [nn.LeakyReLU() , nn.GELU(), nn.ELU()]
# run config
cfg.save_model = True
