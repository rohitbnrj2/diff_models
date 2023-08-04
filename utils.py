import os
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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
    # im = im.resize((256, 256), Image.NEAREST)
    im.save(path)

def plot_loss(args, train_loss: list = None, path = None):
    """
    Plot the training loss for each epoch.
    """

    plt.rcParams["figure.figsize"] = (8.0, 6.0)
    plt.rcParams["font.size"] = 12

    epochs = torch.arange(0, len(train_loss), 1)

    plt.plot(epochs, train_loss)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.close()
    return


def plot_noise_distribution(args, noise, predicted_noise, epoch: int = 0, t: torch.tensor = None):

    plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.title(f"The Histogram is for Timestep {t}")
    plt.legend()
    plt.savefig(f"noise_dist/{args.run_name}/{epoch}.png")
    plt.close()
    return


def get_data(args):
    data_transforms = transforms.Compose([
                 transforms.Resize((args.image_size, args.image_size)),
                 transforms.RandomAdjustSharpness(1.0, p=1),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.num_workers > 16:
        num_workers = 16
    else:
        num_workers = args.num_workers

    dataset = datasets.CIFAR10(root=".", download=True, transform=data_transforms)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("Loss", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("noise_dist", run_name), exist_ok=True)
    os.makedirs(os.path.join("Loss", run_name), exist_ok=True)
    
    return