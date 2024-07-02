# ISIC 2024 - Skin Cancer Detection with 3D-TBP


## Introduction
Skin cancer can be deadly if not caught early, but many populations lack specialized dermatologic care. Over the past several years, dermoscopy-based AI algorithms have been shown to benefit clinicians in diagnosing melanoma, basal cell, and squamous cell carcinoma. However, determining which individuals should see a clinician in the first place has great potential impact. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

Dermatoscope images reveal morphologic features not visible to the naked eye, but these images are typically only captured in dermatology clinics. Algorithms that benefit people in primary care or non-clinical settings must be adept to evaluating lower quality images. This competition leverages 3D TBP to present a novel dataset of every single lesion from thousands of patients across three continents with images resembling cell phone photos.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributors](#contributors)
- [License](#license)

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/phanhoang1803/ISIC2024.git
    cd ISIC2024/src
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

<!-- 3. For pretrained weights:
Please refer to [here](https://github.com/phanhoang1803/ISIC2024/blob/main/README.md#pretrained-weights) -->

## Usage

- Below is script to train the model. For more details about arguments, please refer to `utils/utils.py`.
```sh
python train.py \
    --root_dir path/to/ISIC2024/data/isic-2024-challenge \
    --extra_data_dirs path/to/extra/ISIC/data \
    --achitecture EfficientNet
```

## Results

- Updating ...

## Dependencies

- Python 3.x
- PyTorch
- timm

For a full list of dependencies, see `requirements.txt`.

## Contributors
- phanhoang1803

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


