"""
This script plots the training and testing loss over epochs  
from a CSV file. It is mainly used to visualize the loss evolution  
during model training, with a logarithmic y-scale for better readability.
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('GNN for acceleration/Supervised/Train/Energy/loss_4920.csv')


plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['global_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['test_loss'], label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
