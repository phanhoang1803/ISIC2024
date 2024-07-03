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
from data.data_loading import load_data
from data.dataset import ISICDataset_for_Train, ISICDataset, TBP_Dataset
from data.data_processing import get_transforms, feature_engineering
from models.isic_model import ISICModel
from models.RNN_GRU_model import ISICModel_MaskRNN_GRU
from models.EfficientNet_FPN_SE import EfficientNet_FPN_SE
from models.ensemble_model import EnsembleModel
from models.CombinedAttentionModel import CombinedAttentionModel
# from utils.config import CONFIG
from utils.seed import seed_torch
from utils.utils import make_dirs, save_model
from models.criterion import valid_score, criterion
from utils.utils import parse_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def train_one_epoch(model, optimizer, scheduler, dataloader, meta_feature_columns, device, epoch, CONFIG):
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
    running_auroc  = 0.0
    all_targets = []
    all_outputs = []
    
    # Progress bar for tracking training progress
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for step, data in bar:
        # Move data to device
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        meta_feature_columns = data['meta_feature'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images, meta_feature_columns).squeeze()
        
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
           
        # Backward pass and optimization
        loss.backward()
   
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
        
        # Calculate accuracy
        preds = torch.round(torch.sigmoid(outputs))
        running_corrects += torch.sum(preds == targets.data)
        
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        
        # Update progress bar
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc.item(), LR=optimizer.param_groups[0]['lr'])
   
    # Convert targets and outputs to numpy arrays
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    # Create dataframes for targets and outputs
    solution = pd.DataFrame(all_targets, columns=['target'])
    submission = pd.DataFrame(all_outputs, columns=['target'])
    
    # Calculate AUROC score
    epoch_auroc = valid_score(solution, submission, row_id_column_name='target')
    
    # Update progress bar
    bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc.item(), Train_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])
    
    # Collect garbage
    gc.collect()
    
    return epoch_loss, epoch_auroc, epoch_acc.item()

@torch.inference_mode()
def valid_one_epoch(model, dataloader, meta_feature_columns, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        meta_feature_columns = data['meta_feature'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images, meta_feature_columns).squeeze()
        loss = criterion(outputs, targets)
        
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        # Calculate accuracy
        preds = torch.round(torch.sigmoid(outputs))
        running_corrects += torch.sum(preds == targets.data)
        
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc.item())   
 
    
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    solution = pd.DataFrame(all_targets, columns=['target'])
    submission = pd.DataFrame(all_outputs, columns=['target'])
    
    epoch_auroc = valid_score(solution, submission, row_id_column_name='target')
    
    bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc.item(), Valid_Auroc=epoch_auroc)
    
    gc.collect()
    
    return epoch_loss, epoch_auroc, epoch_acc.item()

def run_training(model, train_loader, valid_loader, meta_feature_columns, optimizer, scheduler, device, num_epochs, CONFIG):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_pauc = -np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss, train_epoch_pauc, train_epoch_acc = train_one_epoch(model, 
                                                                              optimizer=optimizer, 
                                                                              scheduler=scheduler,
                                                                              dataloader=train_loader,
                                                                              meta_feature_columns=meta_feature_columns,
                                                                              device=device, 
                                                                              epoch=epoch, 
                                                                              CONFIG=CONFIG)

        val_epoch_loss, val_epoch_pauc, val_epoch_acc = valid_one_epoch(model, 
                                                                        dataloader=valid_loader, 
                                                                        meta_feature_columns=meta_feature_columns,
                                                                        device=device,
                                                                        epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train pAUC'].append(train_epoch_pauc)
        history['Valid pAUC'].append(val_epoch_pauc)
        history['Train Acc'].append(train_epoch_acc)
        history['Valid Acc'].append(val_epoch_acc)
        history['lr'].append(scheduler.get_lr()[0])

        if best_epoch_pauc <= val_epoch_pauc:
            print(f"Validation pAUC Improved ({best_epoch_pauc} ---> {val_epoch_pauc})")
            best_epoch_pauc = val_epoch_pauc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "pAUC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_pauc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            print("Model Saved")

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, Train pAUC: {train_epoch_pauc:.4f} | "
              f"Valid Loss: {val_epoch_loss:.4f}, Valid Acc: {val_epoch_acc:.4f}, Valid pAUC: {val_epoch_pauc:.4f}")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best pAUC: {:.4f}".format(best_epoch_pauc))

    model.load_state_dict(best_model_wts)

    return model, history

def fetch_scheduler(optimizer, CONFIG):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def prepare_loaders(df, fold, meta_feature_columns, data_transforms, CONFIG):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = TBP_Dataset(df_train, meta_feature_columns=meta_feature_columns, transform=data_transforms["train"])
    valid_dataset = TBP_Dataset(df_valid, meta_feature_columns=meta_feature_columns, transform=data_transforms["valid"])

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
    df = load_data(args.root_dir)
    
    # Perform feature engineering
    df, meta_feature_columns = feature_engineering(df)
    
    # Load additional data if provided
    if args.extra_data_dirs:
        for extra_dir in args.extra_data_dirs:
            extra_df = load_data(extra_dir)
            df = pd.concat([df, extra_df], ignore_index=True)
    
    print("Columns in Final DataFrame:", df.columns)
    print("Sample data from Final DataFrame:\n", df.head())
    
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
        model = CombinedAttentionModel(image_model_name=args.model_name, metadata_dim=len(meta_feature_columns), hidden_dims=[512, 128], metadata_output_dim=128)
    
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
                                  meta_feature_columns=meta_feature_columns,
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