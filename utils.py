import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

matplotlib.use('Agg') 

plt.rcParams["figure.figsize"] = (8.0, 8.0)
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
                 torchvision.transforms.Resize((args.image_size, args.image_size)),
                 torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root=".", download=True, transform=transforms)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    return