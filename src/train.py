import time
import copy
import gc
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from data.data_loading import load_data, downsample
from data.dataset import ISICDataset_for_Train, ISICDataset, TBP_Dataset
from data.data_processing import get_transforms, feature_engineering
from models.isic_model import ISICModel
from models.RNN_GRU_model import ISICModel_MaskRNN_GRU
from models.EfficientNet_FPN_SE import EfficientNet_FPN_SE
from models.ensemble_model import EnsembleModel
from models.CombinedAttentionModel import CombinedAttentionModel
from models.CombinedModel import CombinedModel
# from utils.config import CONFIG
from utils.seed import seed_torch
from utils.utils import make_dirs, save_model
from models.criterion import valid_score, criterion, pAUC_score
from utils.utils import parse_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def train_one_epoch(model, optimizer, scheduler, dataloader, use_meta, device, epoch, CONFIG):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
        dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        device (torch.device): The device where the model and data will be loaded.
        epoch (int): The current epoch number.
        CONFIG (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the average loss and AUROC score for the epoch.
    """
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    # Progress bar for tracking training progress    
    for step, data in enumerate(dataloader):
        # Move data to device
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        if use_meta:
            meta = data['meta'].to(device, dtype=torch.float)
        else:
            meta = None
        
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images, meta).squeeze()
        
        # Calculate loss
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
           
        # Backward pass and optimization
        loss.backward()
   
        # Step the optimizer and scheduler
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
               
        # Collect targets and outputs
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        # Update progress bar
        # bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
   
    # Convert targets and outputs to numpy arrays
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    
    # Calculate pAUC score
    epoch_pauc = pAUC_score(torch.nn.Sigmoid()(torch.from_numpy(all_outputs)), all_targets)
    
    # Update progress bar
    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    # bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_pAUC=epoch_pauc, LR=optimizer.param_groups[0]['lr'])
    
    # Collect garbage
    gc.collect()
    
    return epoch_loss, epoch_pauc

@torch.inference_mode()
def valid_one_epoch(model, dataloader, use_meta, device, epoch):
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): The data loader for the validation data.
        use_meta (bool): Whether to use meta data for validation.
        device (torch.device): The device where the model and data will be loaded.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the average loss and pAUC score for the epoch.
    """
    # Set model to evaluation mode
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    # Progress bar for tracking validation progress
    for step, data in enumerate(dataloader):        
        # Move data to device
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)      
        
        # Move meta data to device if available
        if use_meta:
            meta = data['meta'].to(device, dtype=torch.float) 
        else:
            meta = None
        
        batch_size = images.size(0)

        # Perform forward pass
        outputs = model(images, meta).squeeze()
        loss = criterion(outputs, targets)
        
        # Collect targets and outputs
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        # Update progress bar
        # bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,)   
 
    
    # Convert targets and outputs to numpy arrays
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    # Calculate pAUC score
    epoch_pAUC = pAUC_score(torch.nn.Sigmoid()(torch.from_numpy(all_outputs)), all_targets)
    
    # Update progress bar
    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    # bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_pAUC=epoch_pAUC)
    
    # Collect garbage
    gc.collect()
    
    return epoch_loss, epoch_pAUC

def run_training(model, train_loader, valid_loader, use_meta, optimizer, scheduler, device, num_epochs, CONFIG):
    """
    Trains a model for a given number of epochs using the provided data and hyperparameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        valid_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        use_meta (bool): Whether to use metadata or not.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device where the model and data will be loaded.
        num_epochs (int): The number of epochs to train for.
        CONFIG (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the trained model and the history of training and validation metrics.
    """
    # Check if GPU is available and print GPU information
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()  # Start timer
    best_model_wts = copy.deepcopy(model.state_dict())  # Store initial model weights
    best_epoch_pauc = -np.inf  # Initialize best pAUC score
    history = defaultdict(list)  # Store history of training and validation metrics
    patience = CONFIG['patience']  # Initialize patience counter
    current_patience = 0  # Initialize current patience counter

    print('[INFO] Start training...')
    print('[INFO] Use metadata: {}'.format(use_meta))

    # Train the model for the specified number of epochs
    for epoch in range(1, num_epochs + 1):
        gc.collect()  # Collect garbage

        # Train the model for one epoch
        train_epoch_loss, train_epoch_pauc = train_one_epoch(model, 
                                                             optimizer=optimizer, 
                                                             scheduler=scheduler,
                                                             dataloader=train_loader,
                                                             use_meta=use_meta,
                                                             device=device, 
                                                             epoch=epoch, 
                                                             CONFIG=CONFIG)

        # Validate the model for one epoch
        val_epoch_loss, val_epoch_pauc = valid_one_epoch(model, 
                                                         dataloader=valid_loader, 
                                                         use_meta=use_meta,
                                                         device=device,
                                                         epoch=epoch)

        # Store training and validation metrics in history
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train pAUC'].append(train_epoch_pauc)
        history['Valid pAUC'].append(val_epoch_pauc)
        history['lr'].append(scheduler.get_lr()[0])

        # Update best model weights and pAUC score
        if best_epoch_pauc <= val_epoch_pauc:
            print(f"Validation pAUC Improved ({best_epoch_pauc} ---> {val_epoch_pauc})")
            best_epoch_pauc = val_epoch_pauc
            current_patience = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "pAUC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_pauc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            print("Model Saved")
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Stopping training...')
                break
        
        # Print training and validation metrics
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_epoch_loss:.4f}, Train pAUC: {train_epoch_pauc:.4f} | "
              f"Valid Loss: {val_epoch_loss:.4f}, Valid pAUC: {val_epoch_pauc:.4f}")

    end = time.time()  # End timer
    time_elapsed = end - start  # Calculate elapsed time
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best pAUC: {:.4f}".format(best_epoch_pauc))

    model.load_state_dict(best_model_wts)  # Load best model weights

    return model, history  # Return the trained model and history

def fetch_scheduler(optimizer, CONFIG):
    """
    Fetches the scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        CONFIG (dict): The configuration dictionary.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The scheduler for learning rate adjustment.
            Returns None if no scheduler is specified in the configuration.
    """

    # Check which scheduler is specified in the configuration
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        # Create a CosineAnnealingLR scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        # Create a CosineAnnealingWarmRestarts scheduler
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        # Return None if no scheduler is specified
        return None
    
    return scheduler

def prepare_loaders(df: pd.DataFrame, fold: int, meta_feature_columns: list, data_transforms: dict, CONFIG: dict) -> tuple:
    """
    Prepare data loaders for training and validation datasets.

    Args:
        df (pd.DataFrame): The main dataframe containing the data.
        fold (int): The fold number for validation.
        meta_feature_columns (list): The list of meta feature columns.
        data_transforms (dict): The dictionary containing data transforms.
        CONFIG (dict): The dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the data loaders for training and validation datasets.
    """
    # Handle NaNs if present
    df.fillna(0, inplace=True)  # Replace NaNs with 0 or another appropriate value
    
    # Split the dataframe into training and validation datasets based on the fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    print("[INFO] Train set: ")
    df_train.describe()
    print("[INFO] Valid set: ")
    df_valid.describe()
    
    # Create the datasets
    train_dataset = TBP_Dataset(df_train, meta_feature_columns=meta_feature_columns, transform=data_transforms["train"])
    valid_dataset = TBP_Dataset(df_valid, meta_feature_columns=meta_feature_columns, transform=data_transforms["valid"])

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


def main():
    """
    Main function to run the training process.
    """
    # Parse command line arguments
    args = parse_arguments()
    CONFIG = vars(args)
    print(CONFIG)
    
    # Set seed for reproducibility
    seed_torch(args.seed)

    # Load main data (ISIC 2024)
    print("[INFO] Loading data...")
    df = load_data(args.root_dir, neg_ratio=args.neg_ratio) # Default = -1, load all data
    meta_feature_columns = None
    
    # Load additional data if provided
    if args.extra_data_dirs:
        print("[INFO] Loading additional data...")
        for extra_dir in args.extra_data_dirs:
            extra_df = load_data(extra_dir, neg_ratio=args.extra_neg_ratio) # Default = 0, load only positive samples
            df = pd.concat([df, extra_df], ignore_index=True).reset_index(drop=True)
    
    # if CONFIG['feature_engineering'] == True:
    print("[INFO] Feature Engineering...")
    # Perform feature engineering
    df, meta_feature_columns = feature_engineering(df, use_new_features=CONFIG['use_new_features'])
    
    # Downsample the negative samples
    df = downsample(df, remain_columns=meta_feature_columns, ratio=CONFIG['data_ratio'], seed=CONFIG['seed'], use_clustering=CONFIG['use_clustering'])
    
    print("[INFO] Columns in Final DataFrame:", df.columns)
    print("[INFO] Sample data from Final DataFrame:\n", df.head())
    print("[INFO] Number of positive samples:", df[df['target'] == 1].shape[0])
    print("[INFO] Number of negative samples:", df[df['target'] == 0].shape[0])
    
    # Calculate T_max
    CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
    
    # Create Folds
    sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_ , "kfold"] = int(fold)
    
    # Initialize model
    if CONFIG['architecture'] == 'EfficientNet':
        model = ISICModel(CONFIG['model_name'], pretrained=True, checkpoint_path=CONFIG['checkpoint_path'])
    elif CONFIG['architecture'] == 'MaskRNN_GRU':
        model = ISICModel_MaskRNN_GRU(CONFIG['model_name'], pretrained=True, checkpoint_path=CONFIG['checkpoint_path'])
    elif CONFIG['architecture'] == 'EfficientNet_FPN_SE':
        model = EfficientNet_FPN_SE()
    elif CONFIG['architecture'] == 'EnsembleModel':
        model = EnsembleModel()
    elif CONFIG['architecture'] == 'CombinedAttentionModel':
        model = CombinedAttentionModel(image_model_name=args.model_name, 
                                       metadata_dim=len(meta_feature_columns) if CONFIG['use_meta'] else 0, 
                                       hidden_dims=[512, 128], 
                                       metadata_output_dim=128)
    elif CONFIG['architecture'] == 'CombinedModel':
        model = CombinedModel(image_model_name=args.model_name,
                              metadata_dim=len(meta_feature_columns) if CONFIG['use_meta'] else 0, 
                              hidden_dims=[512, 128], 
                              metadata_output_dim=128,
                              use_attention=args.use_attention,
                              attention_type=args.image_attention_type,
                              num_heads=args.num_heads)
    
    model.to(CONFIG['device'])

    # Prepare data loaders
    transforms = get_transforms(CONFIG)
    train_loader, valid_loader = prepare_loaders(df, fold=CONFIG["fold"], meta_feature_columns=meta_feature_columns, data_transforms=transforms, CONFIG=CONFIG)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, CONFIG)

    # Run training
    model, history = run_training(model, 
                                  train_loader=train_loader, 
                                  valid_loader=valid_loader, 
                                  use_meta=CONFIG['use_meta'],
                                  optimizer=optimizer, 
                                  scheduler=scheduler, 
                                  device=CONFIG['device'], 
                                  num_epochs=CONFIG['epochs'], 
                                  CONFIG=CONFIG)

    # Save history to CSV file
    history = pd.DataFrame.from_dict(history)
    history.to_csv("history.csv", index=False)

if __name__ == "__main__":
    main()
