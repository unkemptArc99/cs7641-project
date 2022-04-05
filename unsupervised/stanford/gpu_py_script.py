import os
import torch
# import cv2
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kornia.feature import DenseSIFTDescriptor, SIFTDescriptor

code_folder = os.getcwd()
annotation_folder = os.path.join(code_folder,'../../dataset/Stanford/Annotation')
images_folder = os.path.join(code_folder,'../../dataset/Stanford/Images')

transform = transforms.Compose([transforms.Resize((255,255)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = datasets.ImageFolder(images_folder, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for images, _ in dataloader:
    SIFT = SIFTDescriptor(patch_size=255)
    descs = SIFT(images)
    print(descs.shape)