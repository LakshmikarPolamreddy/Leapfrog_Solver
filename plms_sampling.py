import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan
import os 

#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from ldm.models.diffusion.plms import PLMSSampler

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

# Initialize the model and sampler
model = get_model()
sampler = PLMSSampler(model)

# Define the classes to be sampled, number of samples per class, and other parameters
start_class = 0
end_class = 999
n_samples_per_class = 50
plms_steps = 4
plms_eta = 0.0
scale = 3.0   # for unconditional guidance

# Create a folder to save the images
output_folder = "/scratch/laks/leapfrog_solver/samples/imagenet/plms_sampling"
os.makedirs(output_folder, exist_ok=True)

# Measure the time for sampling
import time
start_time = time.time()

for class_label in range(start_class, end_class + 1):
    all_samples = []
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
            print(f"Rendering {n_samples_per_class} examples of class '{class_label}' in {plms_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            samples_plms, _ = sampler.sample(S=plms_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, 
                                                 eta=plms_eta)
            x_samples_plms = model.decode_first_stage(samples_plms)
            x_samples_plms = torch.clamp((x_samples_plms+1.0)/2.0, 
                                              min=0.0, max=1.0)
            all_samples.append(x_samples_plms)

    # Combine all samples of the class into a grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = grid.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    # Save the images
    for i, image_data in enumerate(grid):
        image = Image.fromarray(image_data)
        image_filename = f"class_{class_label:03d}_{i:03d}.png"
        image_path = os.path.join(output_folder, image_filename)
        image.save(image_path)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time for sampling: {total_time} seconds")