import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(CONFIG):
    """
    Returns a dictionary of Albumentations transforms for train and valid datasets.

    Args:
        CONFIG (dict): Configuration dictionary containing image size.

    Returns:
        dict: Dictionary containing train and valid Albumentations transforms.
    """
    # Define the train transforms
    train_transforms = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
    
    # Define the valid transforms
    valid_transforms = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
    
    # Return dictionary of transforms
    return {
        "train": train_transforms,
        "valid": valid_transforms
    }
