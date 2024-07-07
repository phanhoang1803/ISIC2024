import pandas as pd
import glob
from sklearn.cluster import KMeans

def downsample(df: pd.DataFrame, remain_columns: list, ratio: int=20, seed: int=42, use_clustering: bool=False):
    # Separate positive and negative samples
    df_positive = df[df['target'] == 1].reset_index(drop=True)
    df_negative = df[df['target'] == 0].reset_index(drop=True)
        
    # Filter df based on negative_ratio
    if ratio == 0:                      # only include positive samples
        df = df_positive
    elif ratio > 0:                     # downsample negative samples
        if use_clustering:                  # downsample benign samples using clustering
            df_negative = downsample_benign_samples(df_negative, sample_count=df_positive.shape[0] * ratio, remain_columns=remain_columns, seed=seed)
        else: 
            df_negative = df_negative.iloc[:df_positive.shape[0] * ratio, :]
            
        # Downsample the negative samples
        df = pd.concat([df_positive, df_negative]).reset_index(drop=True)
    else:                               # load all data
        df = df

    return df



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

    # Check and remove columns containing "Unnamed"
    df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])

    return df

def downsample_benign_samples(df, sample_count, remain_columns, seed):
    """
    Downsample benign samples ensuring diversity by using clustering.
    
    Args:
        benign_df (pd.DataFrame): DataFrame containing benign samples.
        malignant_count (int): Number of malignant samples to match.
        seed (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Downsampled benign samples.
    """
    print("[INFO] Downsample benign samples using clustering")
    
    # Use K-Means clustering to find clusters in benign samples
    num_clusters = sample_count
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    
    # fit on remaining columns
    df['cluster'] = kmeans.fit_predict(df[remain_columns])
    # df['cluster'] = kmeans.fit_predict(df)
    
    # Select one sample from each cluster
    downsampled_benign = df.groupby('cluster').apply(lambda x: x.sample(1, random_state=seed)).reset_index(drop=True)
    
    return downsampled_benign.drop(columns=['cluster'])