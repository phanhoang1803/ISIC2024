import time
import copy
import gc
import numpy as np
import torch
import tqdm
import torch.optim as optim
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from data.data_loading import load_data
from features.dataset import ISICDataset_for_Train, ISICDataset
from features.augmentations import get_transforms
from models.isic_model import ISICModel
# from utils.config import CONFIG
from utils.seed import seed_torch
from utils.utils import make_dirs, save_model
from models.criterion import valid_score, criterion
from utils.utils import parse_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, CONFIG):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    all_targets = []
    all_outputs = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    solution = pd.DataFrame(all_targets, columns=['target'])
    submission = pd.DataFrame(all_outputs, columns=['target'])
    
    epoch_auroc = valid_score(solution, submission, row_id_column_name='target')
    
    bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])
    
    gc.collect()
    
    return epoch_loss, epoch_auroc

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])   
    
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    solution = pd.DataFrame(all_targets, columns=['target'])
    submission = pd.DataFrame(all_outputs, columns=['target'])
    
    epoch_auroc = valid_score(solution, submission, row_id_column_name='target')
    
    bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])
    
    gc.collect()
    
    return epoch_loss, epoch_auroc

def run_training(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs, CONFIG):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_pauc = -np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss, train_epoch_pauc = train_one_epoch(model, optimizer, scheduler,
                                                              dataloader=train_loader,
                                                              device=device, epoch=epoch, CONFIG=CONFIG)

        val_epoch_loss, val_epoch_pauc = valid_one_epoch(model, valid_loader, device=device,
                                                          epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train pAUC'].append(train_epoch_pauc)
        history['Valid pAUC'].append(val_epoch_pauc)
        history['lr'].append(scheduler.get_lr()[0])

        if best_epoch_pauc <= val_epoch_pauc:
            print(f"Validation pAUC Improved ({best_epoch_pauc} ---> {val_epoch_pauc})")
            best_epoch_pauc = val_epoch_pauc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "pAUC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_pauc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            print("Model Saved")

        print()

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

def prepare_loaders(df, fold, data_transforms, CONFIG):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset_for_Train(df_train, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

if __name__ == "__main__":
    args = parse_arguments()
    CONFIG = vars(args)
    print(CONFIG)
    
    seed_torch(args.seed)

    df = load_data(args.root_dir)
    
    CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
    CONFIG['T_max']
    
    # Create Folds
    sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_ , "kfold"] = int(fold)
    
    # Augmentation
    model = ISICModel(CONFIG['model_name'], pretrained=True, checkpoint_path=CONFIG['checkpoint_path'])
    model.to(CONFIG['device'])

    
    transforms = get_transforms(CONFIG)
    train_loader, valid_loader = prepare_loaders(df, fold=CONFIG["fold"], data_transforms=transforms, CONFIG=CONFIG)
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, CONFIG)

    model, history = run_training(model, train_loader, valid_loader, optimizer, scheduler, device=CONFIG['device'], num_epochs=CONFIG['epochs'])

    history = pd.DataFrame.from_dict(history)
    history.to_csv("history.csv", index=False)