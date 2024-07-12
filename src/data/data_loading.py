import pandas as pd
import numpy as np
import glob
from data.data_processing import downsample
import cv2

def load_data(ROOT_DIR, neg_ratio: int=20):
    """
    Load data from the specified ROOT_DIR.

    Args:
        ROOT_DIR (str): The root directory of the data.
        neg_ratio (int): The ratio of negative samples to positive samples.
            If set to 0, only positive samples are included.
            If set to a positive value, negative samples are downsampled.
            If set to a negative value, load all data.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    # Define the train image directory
    TRAIN_DIR = f'{ROOT_DIR}/train-image/image'

    # Get the list of train image file paths
    train_images = sorted(glob.glob(f'{TRAIN_DIR}/*.jpg'))

    # Load the metadata CSV file
    df = pd.read_csv(f'{ROOT_DIR}/train-metadata.csv')

    # Print columns and sample rows for debugging
    print("[INFO] Columns in DataFrame:", df.columns)
    print("[INFO] Sample data from DataFrame:\n", df.head())

    # Ensure 'isic_id' column is present
    if 'isic_id' not in df.columns:
        raise KeyError("Column 'isic_id' is missing from the DataFrame.")

    # Downsample
    df = downsample(df, neg_ratio)

    # Add file path column
    df['file_path'] = df['isic_id'].apply(lambda x: f'{TRAIN_DIR}/{x}.jpg')

    # Ensure 'file_path' column is present
    if 'file_path' not in df.columns:
        raise KeyError("Column 'file_path' is missing from the DataFrame.")

    # Filter the DataFrame based on the valid file paths
    df = df[df['file_path'].isin(train_images)].reset_index(drop=True)

    images = []
    for path in df['file_path']:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            images.append(image)
        else:
            raise ValueError(f"Image at path {path} could not be loaded.")
    df['image_data'] = images

    # Check and remove columns containing "Unnamed"
    df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])

    return df