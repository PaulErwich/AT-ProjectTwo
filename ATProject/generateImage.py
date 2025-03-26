
import torch

from dataclasses import dataclass
from sched import scheduler

from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

from diffusers import DDPMScheduler

from diffusers import UNet2DModel, optimization

@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 5
    eval_batch_size = 4
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 100
    save_model_epochs = 500
    mixed_precision  = "fp16"
    output_dir = "data/testMakingJustImages"

    push_to_hub = False
    hub_model_id = "0"
    hub_private_repo = None
    overwrite_output_dir = True
    seed = 1 # This is used for generation so needs to change?

config = TrainingConfig()


def evaluate(config, epoch, pipeline):
    rSeed = torch.Generator().seed()
    images = pipeline(
        batch_size = config.eval_batch_size,
        num_inference_steps = 500
        ).images

    image_grid = make_image_grid(images, rows=2, cols=2)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir,exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



model = UNet2DModel.from_pretrained("data/32x32UFinalDatasetmk2/unet")
noise_scheduler = DDPMScheduler.from_pretrained("data/32x32UFinalDatasetmk2/scheduler")

pipeline = DDPMPipeline(unet = model, scheduler = noise_scheduler)

for i in range(10):
    evaluate(config, i + 1, pipeline)