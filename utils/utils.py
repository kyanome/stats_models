import yaml
import torch
from torch.utils.data import DataLoader
from models import *
from optimizer import *
from dataset import *
from utils.types_ import *


def setting_optimizer(config) -> List[Any]:
    model = vae_models[config['model_name']['name']](**config['model_params'])
    optimizer_algorithm = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = config['trainer_params']['max_epochs']
    batch_size = config['exp_params']['batch_size']
    dataset = dataset_list[config['exp_params']['dataset']]()
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    model_settings = [model, optimizer_algorithm, data_loader, batch_size, epochs] 
    optimizer = optimizer_list[config['trainer_params']['optimizer']](*model_settings)
    return optimizer

def read_config(yaml_file) -> None:
    with open(yaml_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config