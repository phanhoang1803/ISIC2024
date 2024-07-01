import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
