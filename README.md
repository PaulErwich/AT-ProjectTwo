# AT Project 2
Advanced tech project two. Dungeon layout generation using generative AI

How to run setup and run project:

Python:
Requires a Python 3.12.7 environment, during installation ensure
"Add Python to environment variables" is ticked, otherwise the C++ program
will not be able to run the python script.
Requires the following modules installed via pip:
accelerate
datasets
diffusers
torchvision

This should install all relevant dependancies required to run the project files.

ButterflyTutorial.py is an implementation of the tutorial "Train a diffusion model"
from hugging face

DungeonGenerator.py is the modified tutorial that trains a model on the dungeon
image dataset

generateImage.py is a script designed solely to generate images using a trained model.
A slightly modified version of this file is also included in the C++ project.

