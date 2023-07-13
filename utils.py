import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
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

    return

def plot_noise_distribution(args, noise, predicted_noise, epoch: int = 0):
    plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.legend()
    plt.savefig(f"noise_dist/{args.run_name}/{epoch}.png")
    plt.close()
    return

def get_data(args):
    data_transforms = transforms.Compose([
                 transforms.Resize((args.image_size, args.image_size)),
                 transforms.ToTensor(),
                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 transforms.Lambda(lambda t: (t*2) - 1),
    ])

    if args.num_workers > 16:
        num_workers = 16
    else:
        num_workers = args.num_workers

    dataset = datasets.CIFAR10(root=".", download=True, transform=data_transforms)
    image, _ = dataset[0]
    data = image.repeat(args.batch_size * 4, 1, 1, 1)
    assert len(data.shape) == 4
    dataloader = DataLoader(data, args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("noise_dist", run_name), exist_ok=True)
    return