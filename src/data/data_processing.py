import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def downsample(df: pd.DataFrame, remain_columns: list, ratio: int=20, seed: int=42, down_type: str=None):
    # Separate positive and negative samples
    df_positive = df[df['target'] == 1].reset_index(drop=True)
    df_negative = df[df['target'] == 0].reset_index(drop=True)
        
    # Filter df based on negative_ratio
    if ratio == 0:                      # only include positive samples
        df = df_positive
    elif ratio > 0:                     # downsample negative samples
        if down_type == 'clustering':                  # downsample benign samples using clustering
            df_negative = downsample_benign_samples(df_negative, sample_count=df_positive.shape[0] * ratio, remain_columns=remain_columns, seed=seed)
        elif down_type == 'random':          # downsample benign samples randomly
            df_negative = resample(df_negative, 
                                   replace=False, 
                                   n_samples=df_positive.shape[0] * ratio, 
                                   random_state=seed)
        else: 
            df_negative = df_negative.iloc[:df_positive.shape[0] * ratio, :]
            
        # Downsample the negative samples
        df = pd.concat([df_positive, df_negative]).reset_index(drop=True)
    else:                               # load all data
        df = df

    return df

def resample_data(df: pd.DataFrame, feature_columns: list, target_column: str, upsample_ratio: int = 20, data_ratio: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Upsample positive cases in a DataFrame using SMOTE.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    feature_columns (list): List of column names to be used as features.
    target_column (str): The name of the target column.
    upsample_ratio (int): The ratio of upsampling to perform (default is 20).
    seed (int): Random seed for reproducibility (default is 42).

    Returns:
    pd.DataFrame: The DataFrame with upsampled positive cases.
    """
    # Replace infinity values with NaN
    df[feature_columns].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    df[feature_columns] = imputer.fit_transform(df[feature_columns])

    # Clip large values to prevent issues with clustering
    df[feature_columns] = np.clip(df[feature_columns], -1e9, 1e9)
    
    # Flat the image data
    # df["image_data"] = df["image_data"].apply(lambda x: x.flatten())
    
    # feature_columns = feature_columns + ["image_data"]
    # Separate the features and target
    X = df[feature_columns]
    y = df[target_column]

    # Calculate the number of samples needed for the minority class
    count_negatives = sum(y == 0)
    count_positives = sum(y == 1)
    target_positives = int(count_positives * upsample_ratio)
    target_negatives = int(target_positives * data_ratio)
    
    # Calculate the number of samples needed for the minority class
    smote = SMOTE(sampling_strategy={1: target_positives, 0: min(count_negatives, target_negatives)}, random_state=seed)
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine the resampled features and target into a new DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
    df_resampled[target_column] = y_resampled

    resampled_df = df_resampled.merge(df.drop(columns=feature_columns + [target_column]), left_index=True, right_index=True, how='left')

    return resampled_df

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
    
    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    df[remain_columns] = imputer.fit_transform(df[remain_columns])

    # Clip large values to prevent issues with clustering
    df[remain_columns] = np.clip(df[remain_columns], -1e9, 1e9)

    # Use K-Means clustering to find clusters in samples
    num_clusters = sample_count
    
    pipeline = Pipeline([
        ('kmeans', KMeans(n_clusters=num_clusters, random_state=seed, verbose=True))  # Apply KMeans clustering
    ])

    # Fit on remaining columns
    df['cluster'] = pipeline.fit_predict(df[remain_columns])
    
    # Select one sample from each cluster
    downsampled_samples = df.groupby('cluster').apply(lambda x: x.sample(1, random_state=seed)).reset_index(drop=True)
    
    # Drop the cluster column before returning
    downsampled_samples.drop(columns=['cluster'], inplace=True)
    
    return downsampled_samples

def feature_engineering(df, use_new_features=True):
    """
    Performs feature engineering on the input DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the input data.

    Returns:
        pandas.DataFrame: DataFrame containing the processed data.
        list: List of column names representing the meta-features.
    """
    if use_new_features:
        # Perform feature engineering
        df["age_approx"] = df["age_approx"] / 100  # Normalize age
        
        # New features to try...
        df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
        df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
        df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
        df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
        df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
        df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
        df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
        df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
        df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
        df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
        df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
        df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
        df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
        
        df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
        df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
        df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
        df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
        df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
        df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
        df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
        df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
        df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
        df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
        df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
        df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
        df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
        df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

        # Define the meta-feature columns
        new_num_cols = [
            "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
            "luminance_contrast", "lesion_color_difference", "border_complexity",
            "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
            "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

            "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
            "shape_complexity_index", "color_contrast_index", "log_lesion_area",
            "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
            "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
            "symmetry_perimeter_interaction", "comprehensive_lesion_index",
        ]
        new_cat_cols = ["combined_anatomical_site"]
    else:
        new_num_cols = []
        new_cat_cols = []
        
        # 'tbp_lv_stdLExt', 
    num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
        'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
        'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    ] + new_num_cols
    
    cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"] + new_cat_cols
    
    meta_feature_columns = num_cols + cat_cols
    
    # meta_feature_columns = [
    #     "tbp_lv_z", 
    #     "tbp_lv_nevi_confidence",
    #     "color_uniformity",
    #     "hue_contrast", 
    #     "tbp_lv_Lext",
    #     "tbp_lv_deltaB",
    #     "normalized_lesion_size",
    #     "age_approx",
    #     "3d_position_distance",
    #     "color_contrast_index",
    #     "tbp_lv_deltaA",
    #     "tbp_lv_B",
    #     "tbp_lv_eccentricity",
    #     "tbp_lv_y",
    #     "tbp_lv_Bext",
    #     "lesion_size_ratio",
    #     "size_age_interaction",
    #     "3d_lesion_orientation",
    #     "overall_color_difference",
    #     "lesion_color_difference",
    #     "tbp_lv_x",
    #     "tbp_lv_deltaLBnorm",
    #     "clin_size_long_diam_mm",
    #     "tbp_lv_C",
    #     "mean_hue_difference",
    #     "tbp_lv_Hext",
    #     "tbp_lv_A",
    #     "tbp_lv_symm_2axis",
    #     "lesion_severity_index",
    #     "tbp_lv_symm_2axis_angle"
    # ]
    
    category_encoder = OrdinalEncoder(
        categories='auto',
        dtype=int,
        handle_unknown='use_encoded_value',
        unknown_value=-2,
        encoded_missing_value=-1,
    )

    X_cat = category_encoder.fit_transform(df[cat_cols])
    for c, cat_col in enumerate(cat_cols):
        df[cat_col] = X_cat[:, c]
        
    return df, meta_feature_columns
