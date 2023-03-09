import os.path

import torch
from train import train
from test import test
from SDOBenchmark import SDOBenchmark
from model import model_resnet
import yaml
from pathlib import Path
import torch.nn as nn
import argparse

from torchvision import transforms
from tsne import plot_tsne

parser = argparse.ArgumentParser(description="...")

parser.add_argument("--config_path", default = "./config.yaml", type =str, help = "Path of the config file")
parser.add_argument("--task", default = "plot_tsne", type = str, help = "Type of task")

args = parser.parse_args()

def train_val_split(dataset, val_size = 0.1, seed =55):
    val_size = int(val_size * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(dataset, [train_size, val_size], generator = generator)

def setup(task, config):
    path = Path(config["data"]["path"])
    if task =="train":

        dataset = SDOBenchmark(path / 'training' / 'meta_data.csv', path / 'training', transform = transforms.Compose([
            transforms.Resize(256),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])  )

        # train validation split
        train_dataset, val_dataset =train_val_split(dataset)

        dataloader_tr = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False)
        return dataloader_tr, dataloader_val

    if task == "test":
        dataset = SDOBenchmark(path / 'test' / 'meta_data.csv', path / 'test', transform  = transforms.Compose([

            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]))

        dataloader_ts = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = False)
        return dataloader_ts


def test_command(config):
    ts_dataloader = setup("test", config)
    model = model_resnet(4, 2, 512, 0.2)
    model.eval()
    model.load_state_dict(torch.load(Path(config["path"]) / 'model.pt', map_location=torch.device('cpu')))
    loss  =nn.CrossEntropyLoss()
    device = 'cpu'
    test(ts_dataloader, model, loss, device, Path(config["path"]))

def tsne_command(config):
    ts_dataloader = setup("test", config)
    model = model_resnet(4, 2, 512, 0.2)
    model.eval()
    model.load_state_dict(torch.load(Path(config["path"]) / 'model.pt', map_location=torch.device('cpu')))
    loss = nn.CrossEntropyLoss()
    device = 'cpu'
    plot_tsne(model, ts_dataloader)

def train_command(config):
    tr_dataloader, val_dataloader = setup("train", config )
    model = model_resnet(4, 2, 512, 0.2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    epochs = 300
    epoch_es = 20
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    path = Path(config["path"])
    if not os.path.exists(path):
        os.mkdir(path)
    train(tr_dataloader, val_dataloader, model, loss, epoch_es, epochs, optimizer, device, path  )


if __name__ == '__main__':
    #read configs
    with open(args.config_path) as file:
        parameters = yaml.load(file, Loader = yaml.FullLoader)



    if args.task == "train":

        train_command(parameters)

    if args.task == "test":
        test_command(parameters)

    if args.task == "plot_tsne":
        tsne_command(parameters)














