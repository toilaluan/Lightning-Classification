from torch.utils.data import Dataset
from glob import glob
from clearml import Dataset as Clearml_Dataset
import os
import pandas as pd
from PIL import Image
import timm
import torch


class FlowerDataset(Dataset):
    def __init__(self, root_folder: str, img_size: tuple, is_training=True, model=None):
        self.root_folder = root_folder
        self.data = pd.read_csv(os.path.join(self.root_folder, "annotation.csv"))
        self.data = self.data[self.data.is_train == is_training].reset_index()
        data_config = timm.data.resolve_data_config(model=model)
        data_config["scale"] = (0.8, 1.0)
        data_config["vflip"] = 0.5
        data_config["crop_pct"] = 1.0
        data_config["input_size"] = img_size
        self.normalize = timm.data.create_transform(
            **data_config, is_training=is_training
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rows = self.data.iloc[index]
        image_name = rows["image_name"]
        label = rows["label"]
        ground_truth = int(rows["ground_truth"])
        try:
            image = Image.open(
                os.path.join(self.root_folder, "images", label, image_name)
            ).convert("RGB")
        except:
            return self.__getitem__(index + 1)
        image = self.normalize(image)
        ground_truth = torch.tensor(ground_truth)
        return image, ground_truth


if __name__ == "__main__":
    model = timm.create_model("resnet50", pretrained=False)
    dataset = FlowerDataset(
        img_size=(224, 224),
        is_training=True,
        model=model,
    )
    image, gt = dataset[0]
    print(image.shape, gt)
