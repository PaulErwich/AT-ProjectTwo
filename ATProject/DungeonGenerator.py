from dataclasses import dataclass
from datasets import load_dataset
import accelerate

dataset = load_dataset("imagefolder", data_dir="data/", split="train")

print("setup dataset?")

print(dataset)

@dataclass
class TrainingConfig:
    image_size = 16
    train_batch_size = 6
    eval_batch_size = 4
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision  = "fp16"
    output_dir = "data/generated"

    push_to_hub = False
    hub_model_id = "0"
    hub_private_repo = None
    overwrite_output_dir = True
    seed = 0 # This is used for generation so needs to change?

config = TrainingConfig()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()

fig.show()

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images" : images}

dataset.set_transform(transform)

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

from diffusers import UNet2DModel

print(2 ** (len((224, 448, 672, 896)) - 1))


# Currently number of down/up blocks is reduced because image size is too small
model = UNet2DModel(
    sample_size = config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(16, 32, 64),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        ),
    norm_num_groups = 4
    )

sample_image = dataset[0]["images"].unsqueeze(0)

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Input shape: ", sample_image.shape)
print("Output shape: ", model(sample_image, timestep=0).sample.shape)

