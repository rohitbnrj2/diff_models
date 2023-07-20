import os
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_img(image, path, **kwargs):
    grid = torchvision.utils.make_grid(image, **kwargs)
    ndarr = grid.permute(1,2,0).to("cpu").numpy()
    ndarr = ((ndarr+1)/2)*255
    ndarr = np.clip(ndarr, 0, 255).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)

def plot_noise_distribution(args, noise, predicted_noise, epoch: int = 0, t: torch.tensor = None):

    # Sample data (numpy arrays or lists)
    noise_data = noise.cpu().numpy().flatten()
    predicted_noise_data = predicted_noise.cpu().numpy().flatten()

    # Create a blank image with a white background
    width, height = 400, 300
    background_color = (255, 255, 255)  # White
    image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Create histograms for the data (numpy.histogram returns bins and frequencies)
    bins = 10  
    hist_noise, _ = np.histogram(noise_data, bins=bins, density=True)
    hist_predicted_noise, _ = np.histogram(predicted_noise_data, bins=bins, density=True)

    # Calculate the size of each bar based on the image dimensions
    num_bars = len(hist_noise)
    bar_width = width // num_bars
    
    # Set the scaling factor to fit the data within the image height
    scaling_factor = height / max(max(hist_noise), max(hist_predicted_noise))

    # Draw the histogram bars on the image
    for i in range(num_bars):
        # Normalize the heights based on the scaling factor
        bar_height_noise = int(hist_noise[i] * scaling_factor)
        bar_height_predicted = int(hist_predicted_noise[i] * scaling_factor)

        # Draw the bars for ground truth noise (blue) and predicted noise (red)
        x0, y0 = i * bar_width, height - bar_height_noise
        x1, y1 = (i + 1) * bar_width - 1, height - 1
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 255))  # Blue bars for ground truth noise

        x0, y0 = i * bar_width, height - bar_height_predicted
        x1, y1 = (i + 1) * bar_width - 1, height - 1
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0))  # Red bars for predicted noise

    # Save the image with the histogram-like plot
    image.save(f"noise_dist/{args.run_name}/{epoch}.png")
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
    image, _ = dataset[1]
    image = image.unsqueeze(0)
    assert len(image.shape) == 4
    save_img(image, os.path.join("results", args.run_name, "original.jpg"))
    
    image = image.squeeze(0)
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