import sys
from bayes_opt import BayesianOptimization
import torch 
from torch import nn 

sys.path.append("src")
from Shallow_nn.Shallow_nn import bo_train_shallow_model
from Shallow_nn.Shallow_nn import train_shallow_model
from Config.config import cfg
from dataclasses import dataclass
from Deep_nn.Deep_nn import train_deep_model
from Deep_nn.Deep_nn import bo_train_deep_model


if __name__ == "__main__":
    
    pbounds = {
    'learning_rate': (1e-5, 1e-2),
    'batch_size': (32, 256),
    'output_size': (64, 512),
    'loss_function_idx': (0, 0.99),
    'activation_function_idx': (0, 1.99),
    'epochs': (1,2)
    # Search space for output_size
}
    optimizer = BayesianOptimization(
        f=bo_train_shallow_model,
        pbounds=pbounds,
        random_state=1,
)

# You can now run the optimization
    optimizer.maximize(
        init_points=5,
        n_iter=2
)
    print(optimizer.max)

best_params = optimizer.max['params']
loss_functions = [nn.NLLLoss(), nn.CrossEntropyLoss()]  
loss_fn = loss_functions[int(round(best_params['loss_function_idx']))]

activation_function = [nn.ReLU(),nn.Tanh(),nn.Sigmoid()]
activation_nn = activation_function[int(round(best_params['activation_function_idx']))]


batch_size = int(best_params['batch_size'])
learning_rate = best_params['learning_rate']
output_size = int(best_params['output_size'])
epochs = int(best_params['epochs'])
    
@dataclass
class ModelParams:
    batch_size: int
    learning_rate: float
    output_size: int
    epochs: int
    loss_function: object  
    optimization : str
    save_model : bool
    activation_function: object
    
    
    
cfg = ModelParams(
    batch_size=int(best_params['batch_size']),
    learning_rate=best_params['learning_rate'],
    output_size=int(best_params['output_size']),
    epochs=int(best_params['epochs']),
    loss_function=loss_functions[int(round(best_params['loss_function_idx']))],
    optimization='Adam',
    save_model= True,
    activation_function=activation_function[int(round(best_params['activation_function_idx']))],

    
    
)

train_shallow_model(cfg)
print("Model has been trained successfully...!")

#               ***********************  Deep neural network *******************************************

pbounds = {
    'learning_rate': (1e-5, 1e-2),
    'output_size': (64, 512),
    'batch_size': (32, 256),
    'loss_function_idx': (0, 0.99),
    'activation_function_idx': (0, 1.99),
    'epochs': (1,5)
    # Search space for output_size
}
optimizer = BayesianOptimization(
        f=bo_train_deep_model,
        pbounds=pbounds,
        random_state=1,
)

# You can now run the optimization
optimizer.maximize(
        init_points=5,
        n_iter=2
)
print(optimizer.max)

best_params = optimizer.max['params']
loss_functions = [nn.NLLLoss(), nn.CrossEntropyLoss()]  
loss_fn = loss_functions[int(round(best_params['loss_function_idx']))]
batch_size = int(best_params['batch_size'])
learning_rate = best_params['learning_rate']
output_size = int(best_params['output_size'])
epochs = int(best_params['epochs'])
    
@dataclass
class ModelParams:
    batch_size: int
    learning_rate: float
    output_size: int
    epochs: int
    loss_function: object  
    optimization : str
    save_model : bool
    activation_function: object
    
    
    
cfg = ModelParams(
    batch_size=int(best_params['batch_size']),
    learning_rate=best_params['learning_rate'],
    output_size=int(best_params['output_size']),
    epochs=int(best_params['epochs']),
    loss_function=loss_functions[int(round(best_params['loss_function_idx']))],
    optimization='Adam',
    save_model= True,
    activation_function=activation_function[int(round(best_params['activation_function_idx']))],
)

train_deep_model(cfg)
print("Deep neural model has been trained successfully...!")


#TODO Use early stopping for long Epochs 
#TODO Apply hyperparamether Optimization for finding the best hyperparameters of the model.
