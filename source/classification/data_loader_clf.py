import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    CenterCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
)
import pandas as pd
from PIL import Image
import os


def image_transformation(image, cfg: dict):
    trans_img = image.copy()
    transformer = Compose(
        [
            Resize(
                size=(
                    cfg["transforms"]["resize"]["width"],
                    cfg["transforms"]["resize"]["height"],
                ),
                interpolation=Image.BILINEAR,
            ),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=cfg["transforms"]["randomrotation"]["degrees"]),
            ColorJitter(
                brightness=cfg["transforms"]["colorjitter"]["brightness"],
                contrast=cfg["transforms"]["colorjitter"]["contrast"],
                saturation=cfg["transforms"]["colorjitter"]["saturation"],
                hue=cfg["transforms"]["colorjitter"]["hue"],
            ),
            CenterCrop(size=(cfg["transforms"]["cent_crop"]["size"])),
            ToTensor(),
            Normalize(
                mean=cfg["transforms"]["normalize"]["mean"],
                std=cfg["transforms"]["normalize"]["std"],
            ),
        ]
    )
    return transformer(trans_img)


class ClassifierData(Dataset):
    def __init__(self, cfg, label_cfg, is_train: bool, transforms: bool):
        super().__init__()
        if is_train:
            self.image_folder = cfg["data"]["train_images"]
            self.data = pd.read_csv(cfg["data"]["train_labels"])
        else:
            self.image_folder = cfg["data"]["val_images"]
            self.data = pd.read_csv(cfg["data"]["val_labels"])

        self.transforms = transforms
        self.cfg = cfg
        self.label_cfg = label_cfg

    def __len__(self):
        return len(os.listdir(self.image_folder))

    def __getitem__(self, index):
        image_name = self.data.loc[index, "file_name"]
        image = cv2.imread(os.path.join(self.image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transforms:
            image = image_transformation(image, self.cfg)
        if image_name:
            age = self.data.loc[index, "age"]
            race = self.data.loc[index, "race"]
            masked = self.data.loc[index, "masked"]
            skintone = self.data.loc[index, "skintone"]
            emotion = self.data.loc[index, "emotion"]
            gender = self.data.loc[index, "gender"]
            # info = {
            #     "image_name": image_name,
            #     "age": self.label_cfg["age"][age],
            #     "race": self.label_cfg["race"][race],
            #     "masked": self.label_cfg["masked"][masked],
            #     "skintone": self.label_cfg["skintone"][skintone],
            #     "emotion": self.label_cfg["emotion"][emotion],
            #     "gender": self.label_cfg["gender"][gender],
            # }
            info = torch.Tensor(
                [
                    self.label_cfg["age"][age],
                    self.label_cfg["gender"][gender],
                    self.label_cfg["emotion"][emotion],
                ]
                
            )
            return image, info
        else:
            info = {"image_name": image_name}
            return image, info


class Get_Loader:
    def __init__(self, cfg, label_cfg):
        self.batch_size = cfg["data"]["batch_size"]
        self.cfg = cfg
        self.label_cfg = label_cfg

    def load_train_val(self):
        print("Loading train_val data...")

        train_data = ClassifierData(
            cfg=self.cfg, label_cfg=self.label_cfg, is_train=True, transforms=True
        )
        train_loader = DataLoader(
            train_data, self.batch_size, shuffle=True, num_workers=4, drop_last=True
        )

        val_data = ClassifierData(
            cfg=self.cfg, label_cfg=self.label_cfg, is_train=False, transforms=True
        )
        val_loader = DataLoader(
            val_data, self.batch_size, shuffle=True, num_workers=4, drop_last=False
        )
        print("Loading completely...")
        return train_loader, val_loader
