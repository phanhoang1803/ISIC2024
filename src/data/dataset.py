import cv2
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(config):
    """
    Returns a dictionary of Albumentations transforms for train and valid datasets.

    Args:
        config (dict): Configuration dictionary containing image size.

    Returns:
        dict: Dictionary containing train and valid Albumentations transforms.
    """
    # Define the transforms
    image_size = config['img_size']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    max_pixel_value = 255.0

    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0),
        ToTensorV2()], p=1.)

    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0),
        ToTensorV2()], p=1.)

    return {
        "train": train_transform,
        "valid": valid_transform
    }
    
    
class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }


class TBP_Dataset(Dataset):
    def __init__(self, df, meta_feature_columns, transform=None, target_to_prob=False):
        self.df = df.reset_index(drop=True)
        self.meta_feature_columns = meta_feature_columns
        self.transform = transform

        if target_to_prob:
            df_positive = df[df["target"] == 1].reset_index()
            df_negative = df[df["target"] == 0].reset_index()

            # 0 -> (100 - confidence) * 0.5 / 100
            # 1 -> confidence * 0.5 / 100 + 0.5
            df_positive["target"] = df_positive["tbp_lv_dnn_lesion_confidence"] * 0.5 / 100.0 + 0.5
            df_negative["target"] = (100.0 - df_negative["tbp_lv_dnn_lesion_confidence"]) * 0.5 / 100.0
    
            df = pd.concat([df_positive, df_negative]).reset_index()
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        image = cv2.imread(row['file_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = row['target']
        
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.meta_feature_columns is not None:
            # Load meta data and fill missing values
            meta = row[self.meta_feature_columns].values.astype(np.float32)
            meta = np.nan_to_num(meta)
            meta = torch.tensor(meta, dtype=torch.float)
            # print("[INFO] meta in TBP_Dataset:", meta)
            
            return {
                'image': image,
                'target': target,
                'meta': meta
            }
        else:
            return {
                'image': image,
                'target': target
            }