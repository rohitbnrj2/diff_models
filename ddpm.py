import os
import logging
import argparse
import multiprocessing
import torch
import torch.nn as nn
import matplotlib
from tqdm import tqdm
from torch import optim
from utils import *
from unet import UNet
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')


class Diffusion:

    def __init__(self, timesteps: int = 1000, beta_start: float = 2e-4, beta_end: float = 2e-2, 
                 img_size: int = 64, device: str = "cpu"):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.betas = self.noise_scheduler()
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def noise_scheduler(self, sch_type: str = "linear"):
        if sch_type == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
            return betas
        
        elif sch_type == "cosine":
            s = 8e-2
            betas = self.betas_for_alpha_bar(lambda t: torch.cos((t+s)/(1+s))*torch.pi/2)**2
            return betas
        else:
            print(f"Invalid scheduler type {sch_type}")
    
    def betas_for_alpha_bar(self, alpha_bar, max_beta : float = 0.999):
        # Don't understand the code for cosine scheduling, especially this function
        betas = []
        for i in range(self.timesteps):
            t1 = torch.tensor(i/self.timesteps)
            t2 = torch.tensor((i+1)/self.timesteps)             # look into why (i+1) all the way to timesteps.
            betas.append(torch.min((1-alpha_bar(t2)/alpha_bar(t1)), torch.tensor(max_beta)))
        
        betas = torch.tensor(betas)
        return
    
    def noise_images(self, x, t):
        sqrt_alphas_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alphas_bar = torch.sqrt(1-self.alpha_bar[t])[:, None, None, None]
        eps = torch.randn_like(x)
        noised_img = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * eps
        return noised_img, eps
    
    def sample_timestep(self, n: int = 1):
        t = torch.randint(low=1, high=self.timesteps, size=(n,))
        return t
    
    def sample(self, model: nn.Module = None, n: int = 1):
        logging.info(f"Sampling {n} new images")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size))
            for i in range(self.timesteps)[::-1]:
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alphas = self.alphas[t][:, None, None, None]
                alphas_bar = self.alpha_bar[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1/torch.sqrt(alphas) * (x - ((1-alphas)/(torch.sqrt(1-alphas_bar))) * predicted_noise) + torch.sqrt(betas) * noise
        
        model.train()
        x = x.clamp(-1, 1)
        x = ((x + 1)/2) * 255
        x = x.type(torch.uint8)
        return x
    
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    for epoch in range(args.epoch):
        logging.info(f"Starting epoch {epoch}")
        pbar = tqdm(dataloader)
        for i, (images) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE = loss.item())
            logger.add_scalar("MSE", loss.item(), global_step= epoch * l + i)
        
        sampled_images = diffusion.sample(model, n = images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epoch = 5
    args.batch_size = 4
    args.image_size = 64
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 1e-4

    train(args)

if __name__ == "__main__":
    launch()