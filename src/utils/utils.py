import os
import shutil
import torch

# Import parser
from argparse import ArgumentParser

def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
def save_model(model, model_name, path):
    model_path = f'{path}/{model_name}.pth'
    torch.save(model.state_dict(), model_path)

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--architecture', type=str, default="EfficientNet", help='Architecture')
    parser.add_argument('--root_dir', type=str, required=True,help='Root directory of data') # Required
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path') 

    ### Additional data
    parser.add_argument('--extra_data_dirs', type=str, nargs='*', help='List of additional directories containing training data')  
    
    ### CONFIG
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=384, help='Image size')
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b0_ns', help='Model name')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Train batch size')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='Valid batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='Scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--T_max', type=int, default=500, help='T_max for CosineAnnealingLR')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--n_accumulate', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--n_fold', type=int, default=5, help='Number of folds')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    return args
