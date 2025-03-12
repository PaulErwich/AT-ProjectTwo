from dataclasses import dataclass
from sched import scheduler
from datasets import load_dataset
import accelerate
from torch.optim import lr_scheduler

dataset = load_dataset("imagefolder", data_dir="data/TrainingData/32x32Uniform", split="train")

print("setup dataset?")

print(dataset)

@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 4
    eval_batch_size = 4
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 100
    save_model_epochs = 500
    mixed_precision  = "fp16"
    output_dir = "data/32x32UNoGenNoFlipNAdam"

    push_to_hub = False
    hub_model_id = "0"
    hub_private_repo = None
    overwrite_output_dir = True
    seed = 1 # This is used for generation so needs to change?

config = TrainingConfig()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()

fig.show()

from torchvision import transforms

# Random rotations
# Random flips to help "increase" data set
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )

def transform(examples):
    images = [preprocess(image.convert("L")) for image in examples["image"]]
    return {"images" : images}

dataset.set_transform(transform)

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

from diffusers import UNet2DModel, optimization

print(2 ** (len((224, 448, 672, 896)) - 1))


# Currently number of down/up blocks is reduced because image size is too small
model = UNet2DModel(
    sample_size = config.image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(config.image_size, config.image_size * 2, config.image_size * 2, config.image_size * 4),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        ),
    norm_num_groups = 8
    )

sample_image = dataset[0]["images"].unsqueeze(0)

#for param_tensor in model.state_dict():
   # print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Input shape: ", sample_image.shape)
print("Output shape: ", model(sample_image, timestep=0).sample.shape)

from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=500, variance_type="fixed_small")
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([5])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# No permute necessary compared to tutorial because it's only 2 dimensions
# It doesn't need channel number as I'm using greyscale
Image.fromarray(((noisy_image + 1.0) * 127.5).type(torch.uint8).numpy()[0][0], mode="L")

import torch.nn.functional as F

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.NAdam(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

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

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(config: TrainingConfig, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision = config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir = os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            pass
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
        )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable = not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype = torch.int64
                )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict = False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr":lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler = noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

pipeline = DDPMPipeline(unet = model, scheduler = noise_scheduler)

for i in range(10):
    evaluate(config, i, pipeline)

print("Hello at end")

