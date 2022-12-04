import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class RetinaDataset(Dataset):
    def __init__(self, labels, data_dir, label_num, transform=None):
        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform
        self.label_num = label_num  # this is the index of the condition predicted for

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, str(self.labels.iloc[idx, 0]) + ".png")
        image = read_image(img_path)
        label = self.labels.iloc[idx, self.label_num]
        if self.transform:
            image = self.transform(image)
        return image, label
