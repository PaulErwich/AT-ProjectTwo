# AT Project 2
Advanced tech project two. Dungeon layout generation using generative AI

How to setup and run project:

Requires Visual studio 2022/n
Need the Python development and Desktop development with C++ modules
for visual studio installed

Python:

Requires a Python 3.12.7
https://www.python.org/downloads/release/python-3127/
Install "Windows Installer (64-bit)" from bottom of the page

Select custom installation and ensure "Add Python to environment variables" 
is ticked, otherwise the C++ program will not be able to run the python script.

Create virtual environment
Requires the following modules installed via pip:
accelerate
datasets
diffusers
torchvision

For base version 3.12.7
torch
diffusers
accelerate

This should install all relevant dependancies required to run the project files.

ButterflyTutorial.py is an implementation of the tutorial "Train a diffusion model"
from hugging face

DungeonGenerator.py is the modified tutorial that trains a model on the dungeon
image dataset

generateImage.py is a script designed solely to generate images using a trained model.
A slightly modified version of this file is also included in the C++ project.

C++:
When opening visual studio:
	Select "Open a local folder"
	Open the CMakeProject1 folder

As the project is CMake based, Visual studio will automatically sort the compiler
then run the project.

Keybinds:
	G - generate 100 dungeon layout images using Python
	V - validate generated images and sort them into relevant folders
	B - build dungeon using premade layout
	R - procedurally generate dungeon layout and build dungeon with it
