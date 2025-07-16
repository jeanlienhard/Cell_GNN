"""
This script reads a CSV file containing the training history of losses from a GNN model 
trained to model an energy with perimeter.  
It samples the data every 10 epochs to make the plot clearer, 
then plots the training and test losses on a logarithmic scale. 
The plot helps visualize how the losses evolve over the training process.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/Training/Energy_with_perimeter/Energy_with_perimeterloss_400.csv')


df_sampled = df.iloc[::10, :]

plt.figure(figsize=(10, 6))
plt.plot(df_sampled['epoch'], df_sampled['global_loss'], label='Train Loss', marker='o')
plt.plot(df_sampled['epoch'], df_sampled['test_loss'], label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
