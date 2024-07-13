import json
import re
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image
import numpy as np


class Fractal_Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.root = root
        self.Transform = transform
        self.data_list_name = "train.json"
        with open(os.path.join(root, self.data_list_name), 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)


    def read_image_names(self, file_path):
        image_names = []
        with open(file_path, 'r') as file:
            for line in file:
                # Remove newline characters and add the image name to the list
                image_names.append(line.strip())
        return image_names

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        image_name = self.data_list[item]
        image_path = os.path.join(self.root, image_name)

        img = Image.open(image_path)
        # Ensure the image is in RGB format
        img = img.convert('RGB')
        # Resize the image
        img = img.resize((256, 256))
        # Convert the image to a NumPy array
        img_array = np.array(img)
        # Transpose the array to match the original code
        img_array = img_array.transpose(2, 0, 1)
        img = torch.FloatTensor(img_array/255.0)
        if self.Transform is not None:
            image = self.Transform(img)
        meta = {'image': image}
        return meta


def get_dataDB_loader(data_folder, batch_size=16):
    normalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    train_dataset = Fractal_Dataset(root=data_folder, transform=normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return train_loader

