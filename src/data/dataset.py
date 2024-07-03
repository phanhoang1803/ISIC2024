import cv2
import random
import torch
from torch.utils.data import Dataset
import numpy as np


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
    def __init__(self, df, meta_feature_columns, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_feature_columns = meta_feature_columns
        self.transform = transform
    
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