from glob import glob
import os
import sys
import numpy as np

sys.path.append("..")
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import PIL.Image as pil_img


class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        img_dir = "data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images"
        self.paths = glob(os.path.join(img_dir, "*/*.jpg"))
        labels = []

        for p in self.paths:
            label = p.split("/")[-2]
            labels.append(label)
        # Implement additional initialization logic if needed

        self.labels = LabelEncoder().fit_transform(labels)

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.paths)

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        image = np.array(pil_img.open(self.paths[index]).convert("RGB"))
        label = self.labels[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
