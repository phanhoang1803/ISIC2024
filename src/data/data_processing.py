import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
        
    num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
        'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
        'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    ] + new_num_cols
    
    cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"] + new_cat_cols
    
    meta_feature_columns = num_cols + cat_cols
    
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
