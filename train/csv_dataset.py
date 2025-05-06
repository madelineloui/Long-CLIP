import csv
import PIL
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

class CSVValDataset(data.Dataset):
    def __init__(self, csv_path, image_root="", max_items=1000):
        self.image_root = image_root
        self.max_items = max_items

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = [row for i, row in enumerate(reader) if i < self.max_items]

        _, self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image_path = os.path.join(self.image_root, row['filepath'])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        caption = row['long caption'].replace("\n", " ")
        return image_tensor, caption


class CSVTrainDataset(data.Dataset):
    def __init__(self, csv_path, image_root=""):
        self.image_root = image_root

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

        _, self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image_path = os.path.join(self.image_root, row['filepath'])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        caption = row['long caption'].replace("\n", " ")
        caption_short = row['short caption'].replace("\n", " ")
        return image_tensor, caption, caption_short
