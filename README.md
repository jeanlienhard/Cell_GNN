# Cell_GNN

This directory contains the code corresponding to my master's project, which seeks to use GNNs to model cell movements.

## Content

The *Data* folder contains the training data. The raw data corresponds to the simulation data and the random data corresponds to completely randomly generated positions.

The *GNN for accelertion* folder is used to train a GNN to predict acceleration, either with supervised or unsupervised training, depending on the subfolder.

The *GNN for energy* folder is used to train a GNN to predict the system's potential energy.

And the *Results processing* folder is used to measure and visualize model results.

For GNN training, the gnn_model_etc file contains the GNN itself, the global_model_etc file contains pre-processing and post-processing, and the optimise_train_etc file is used for training.