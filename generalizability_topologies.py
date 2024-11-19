#!/usr/bin/env python3

import LSTM_model as mine
import coloredPrinting as pr
import seqMaker
import aggregate
import maxDepth as depth
import observableToSink as obs
import LSTM_FED
import pandas as pd
import numpy as np
import os
import time
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import csv
import argparse



"""
In this scenario we have 3 IDS models, trained with the shallow, medium and deep data.
We test the generalizability of each model to detect the other attack types.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Generalizability - Network Topology: ")

# Add the arguments
parser.add_argument('-lr', type=float, default = 0.001, help = 'Learning rate')
parser.add_argument('-ep', type= int, default = 1, help = 'Number of Epochs')
parser.add_argument('-run', type= int, default = 1, help = 'Number of Runs')
parser.add_argument('-batch',type = int, default = 2048, help = 'Batch Size')
parser.add_argument('-add',type = str, default = '/Path/to/...', help = 'Save Address')

# Parse the arguments
args = parser.parse_args()

# Read the arguments
lr = args.lr                     # learninng rate in gradient descent
epochs = args.ep                 # number of epochs
runs = args.run                  # number of runs, to generate a boxplot we trained and tested each model 10 times
batch_size = args.batch          # batch size
add = args.add                   # address to save the results

#Defining global variables
input_dim = 14                   # 7 fearures: mean of (Rank, DIS-s, DIS-r, DIO-s, DIO-r, DAO-r, tots) + 
                                 # 7 features: standard deviation of (Rank, DIS-s, DIS-r, DIO-s, DIO-r, DAO-r, tots)
hidden_dim = 10                  # LSTM hyperparameter
fc_hidden_dim = 10               # LSTM hyperparameter
num_layers = 1                   # LSTM hyperparameter
output_dim = 2                   # output: attack, non-attack
sequence_length = 10             # sequence length (10 min)
dropout_rate = 0.5



roc_auc_shallow_shallow = []     # records ROC-AUC of a model that is trained in a shallow network and tested in a shallow network
roc_auc_shallow_medium = []      # records ROC-AUC of a model that is trained in a shallow network and tested in a medium network
roc_auc_shallow_deep = []        # records ROC-AUC of a model that is trained in a shallow network and tested in a deep network
roc_auc_medium_shallow = []      # ...
roc_auc_medium_medium = []
roc_auc_medium_deep = []
roc_auc_deep_shallow = []
roc_auc_deep_medium = []
roc_auc_deep_deep = []




for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   shallow_Train, shallow_Test = aggregate.aggregate_list_shallow()
   medium_Train, medium_Test = aggregate.aggregate_list_medium()
   deep_Train, deep_Test = aggregate.aggregate_list_deep()


   pr.prGreen(len(shallow_Train))
   pr.prGreen(len(shallow_Test))
   pr.prRed(len(medium_Train))
   pr.prRed(len(medium_Test))
   pr.prGreen(len(deep_Train))
   pr.prGreen(len(deep_Test))
   print(".................................")
   print("Normalize SHALLOW: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in shallow_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   shallow_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   shallow_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   shallow_max = shallow_all_maxs_df.max(axis = 0)
   shallow_min = shallow_all_mins_df.min(axis = 0)
   
   shallow_normalized_dfs_train = [df.apply(lambda x: (x - shallow_min[x.name]) / (shallow_max[x.name] - shallow_min[x.name])) for df in shallow_Train]
   

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   print(".................................")
   print("Normalize MEDIUM: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in medium_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   medium_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   medium_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   medium_max = medium_all_maxs_df.max(axis = 0)
   medium_min = medium_all_mins_df.min(axis = 0)
   medium_normalized_dfs_train = [df.apply(lambda x: (x - medium_min[x.name]) / (medium_max[x.name] - medium_min[x.name])) for df in medium_Train]
   

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".................................")
   print("Normalize DEEP : ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in deep_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   deep_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   deep_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   deep_max = deep_all_maxs_df.max(axis = 0)
   deep_min = deep_all_mins_df.min(axis = 0)
   deep_normalized_dfs_train = [df.apply(lambda x: (x - deep_min[x.name]) / (deep_max[x.name] - deep_min[x.name])) for df in deep_Train]
   

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   shallow_normalized_dfs_test_shallow = [df.apply(lambda x: (x - shallow_min[x.name]) / (shallow_max[x.name] - shallow_min[x.name])) for df in shallow_Test]
   shallow_normalized_dfs_test_medium = [df.apply(lambda x: (x - shallow_min[x.name]) / (shallow_max[x.name] - shallow_min[x.name])) for df in medium_Test]
   shallow_normalized_dfs_test_deep = [df.apply(lambda x: (x - shallow_min[x.name]) / (shallow_max[x.name] - shallow_min[x.name])) for df in deep_Test]

   medium_normalized_dfs_test_shallow = [df.apply(lambda x: (x - medium_min[x.name]) / (medium_max[x.name] - medium_min[x.name])) for df in shallow_Test]
   medium_normalized_dfs_test_medium = [df.apply(lambda x: (x - medium_min[x.name]) / (medium_max[x.name] - medium_min[x.name])) for df in medium_Test]
   medium_normalized_dfs_test_deep = [df.apply(lambda x: (x - medium_min[x.name]) / (medium_max[x.name] - medium_min[x.name])) for df in deep_Test]

   deep_normalized_dfs_test_shallow = [df.apply(lambda x: (x - deep_min[x.name]) / (deep_max[x.name] - deep_min[x.name])) for df in shallow_Test]
   deep_normalized_dfs_test_medium = [df.apply(lambda x: (x - deep_min[x.name]) / (deep_max[x.name] - deep_min[x.name])) for df in medium_Test]
   deep_normalized_dfs_test_deep = [df.apply(lambda x: (x - deep_min[x.name]) / (deep_max[x.name] - deep_min[x.name])) for df in deep_Test]
   # ------------------------------------
   # make them all seq and the concatenate
   sequencedshallow_Train = [seqMaker.seq_maker(df,10) for df in shallow_normalized_dfs_train]
   sequencedshallow_Test_shallow = [seqMaker.seq_maker(df,10) for df in shallow_normalized_dfs_test_shallow]
   sequencedshallow_Test_medium = [seqMaker.seq_maker(df,10) for df in shallow_normalized_dfs_test_medium]
   sequencedshallow_Test_deep = [seqMaker.seq_maker(df,10) for df in shallow_normalized_dfs_test_deep]

   sequencedmedium_Train = [seqMaker.seq_maker(df,10) for df in medium_normalized_dfs_train]
   sequencedmedium_Test_shallow = [seqMaker.seq_maker(df,10) for df in medium_normalized_dfs_test_shallow]
   sequencedmedium_Test_medium = [seqMaker.seq_maker(df,10) for df in medium_normalized_dfs_test_medium]
   sequencedmedium_Test_deep = [seqMaker.seq_maker(df,10) for df in medium_normalized_dfs_test_deep]

   sequenceddeep_Train = [seqMaker.seq_maker(df,10) for df in deep_normalized_dfs_train]
   sequenceddeep_Test_shallow = [seqMaker.seq_maker(df,10) for df in deep_normalized_dfs_test_shallow]
   sequenceddeep_Test_medium = [seqMaker.seq_maker(df,10) for df in deep_normalized_dfs_test_medium]
   sequenceddeep_Test_deep = [seqMaker.seq_maker(df,10) for df in deep_normalized_dfs_test_deep]


   sequencedshallow_Train = pd.concat(sequencedshallow_Train, ignore_index=True)
   sequencedshallow_Test_shallow = pd.concat(sequencedshallow_Test_shallow, ignore_index=True)
   sequencedshallow_Test_medium = pd.concat(sequencedshallow_Test_medium, ignore_index=True)
   sequencedshallow_Test_deep = pd.concat(sequencedshallow_Test_deep, ignore_index=True)

   sequencedmedium_Train = pd.concat(sequencedmedium_Train, ignore_index=True)
   sequencedmedium_Test_shallow = pd.concat(sequencedmedium_Test_shallow, ignore_index=True)
   sequencedmedium_Test_medium = pd.concat(sequencedmedium_Test_medium, ignore_index=True)
   sequencedmedium_Test_deep = pd.concat(sequencedmedium_Test_deep, ignore_index=True)

   sequenceddeep_Train = pd.concat(sequenceddeep_Train, ignore_index=True)
   sequenceddeep_Test_shallow = pd.concat(sequenceddeep_Test_shallow, ignore_index=True)
   sequenceddeep_Test_medium = pd.concat(sequenceddeep_Test_medium, ignore_index=True)
   sequenceddeep_Test_deep = pd.concat(sequenceddeep_Test_deep, ignore_index=True)


   # extract X and y for shallow, medium, deep
   Xshallow_Train = sequencedshallow_Train.iloc[:, :-1].values  # All columns except the last one
   yshallow_Train = sequencedshallow_Train.iloc[:, -1].values  # The last column
   Xmedium_Train = sequencedmedium_Train.iloc[:, :-1].values  # All columns except the last one
   ymedium_Train = sequencedmedium_Train.iloc[:, -1].values  # The last column
   Xdeep_Train = sequenceddeep_Train.iloc[:, :-1].values  # All columns except the last one
   ydeep_Train = sequenceddeep_Train.iloc[:, -1].values  # The last column

   Xshallow_Test_shallow = sequencedshallow_Test_shallow.iloc[:, :-1].values  # All columns except the last one
   yshallow_Test_shallow = sequencedshallow_Test_shallow.iloc[:, -1].values  # The last column
   Xshallow_Test_medium = sequencedshallow_Test_medium.iloc[:, :-1].values  # All columns except the last one
   yshallow_Test_medium = sequencedshallow_Test_medium.iloc[:, -1].values  # The last column
   Xshallow_Test_deep = sequencedshallow_Test_deep.iloc[:, :-1].values  # All columns except the last one
   yshallow_Test_deep = sequencedshallow_Test_deep.iloc[:, -1].values  # The last column

   Xmedium_Test_shallow = sequencedmedium_Test_shallow.iloc[:, :-1].values  # All columns except the last one
   ymedium_Test_shallow = sequencedmedium_Test_shallow.iloc[:, -1].values  # The last column
   Xmedium_Test_medium = sequencedmedium_Test_medium.iloc[:, :-1].values  # All columns except the last one
   ymedium_Test_medium = sequencedmedium_Test_medium.iloc[:, -1].values  # The last column
   Xmedium_Test_deep = sequencedmedium_Test_deep.iloc[:, :-1].values  # All columns except the last one
   ymedium_Test_deep = sequencedmedium_Test_deep.iloc[:, -1].values  # The last column

   Xdeep_Test_shallow = sequenceddeep_Test_shallow.iloc[:, :-1].values  # All columns except the last one
   ydeep_Test_shallow = sequenceddeep_Test_shallow.iloc[:, -1].values  # The last column
   Xdeep_Test_medium = sequenceddeep_Test_medium.iloc[:, :-1].values  # All columns except the last one
   ydeep_Test_medium = sequenceddeep_Test_medium.iloc[:, -1].values  # The last column
   Xdeep_Test_deep = sequenceddeep_Test_deep.iloc[:, :-1].values  # All columns except the last one
   ydeep_Test_deep = sequenceddeep_Test_deep.iloc[:, -1].values  # The last column



   # tensorizing the data for shallow
   # ----------------------------------------
   Xshallow_Train = np.array(Xshallow_Train)
   yshallow_Train = np.array(yshallow_Train)
   Xshallow_Test_shallow = np.array(Xshallow_Test_shallow)
   yshallow_Test_shallow = np.array(yshallow_Test_shallow)
   Xshallow_Test_medium = np.array(Xshallow_Test_medium)
   yshallow_Test_medium = np.array(yshallow_Test_medium)
   Xshallow_Test_deep = np.array(Xshallow_Test_deep)
   yshallow_Test_deep = np.array(yshallow_Test_deep)

   Xshallow_Train = torch.tensor(Xshallow_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   yshallow_Train = torch.tensor(yshallow_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xshallow_Test_shallow = torch.tensor(Xshallow_Test_shallow, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yshallow_Test_shallow = torch.tensor(yshallow_Test_shallow, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xshallow_Test_medium = torch.tensor(Xshallow_Test_medium, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yshallow_Test_medium = torch.tensor(yshallow_Test_medium, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xshallow_Test_deep = torch.tensor(Xshallow_Test_deep, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yshallow_Test_deep = torch.tensor(yshallow_Test_deep, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xshallow_Train = torch.nan_to_num(Xshallow_Train, nan=0.0)
   #yshallow_Train = torch.nan_to_num(yshallow_Train, nan=0.0)
   Xshallow_Test_shallow = torch.nan_to_num(Xshallow_Test_shallow, nan=0.0)
   Xshallow_Test_medium = torch.nan_to_num(Xshallow_Test_medium, nan=0.0)
   Xshallow_Test_deep = torch.nan_to_num(Xshallow_Test_deep, nan=0.0)
   #yshallow_Test = torch.nan_to_num(yshallow_Test, nan=0.0)

   # tensorizing the data for medium
   # ----------------------------------------
   Xmedium_Train = np.array(Xmedium_Train)
   ymedium_Train = np.array(ymedium_Train)
   Xmedium_Test_shallow = np.array(Xmedium_Test_shallow)
   ymedium_Test_shallow = np.array(ymedium_Test_shallow)
   Xmedium_Test_medium = np.array(Xmedium_Test_medium)
   ymedium_Test_medium = np.array(ymedium_Test_medium)
   Xmedium_Test_deep = np.array(Xmedium_Test_deep)
   ymedium_Test_deep = np.array(ymedium_Test_deep)

   Xmedium_Train = torch.tensor(Xmedium_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ymedium_Train = torch.tensor(ymedium_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xmedium_Test_shallow = torch.tensor(Xmedium_Test_shallow, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ymedium_Test_shallow = torch.tensor(ymedium_Test_shallow, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xmedium_Test_medium = torch.tensor(Xmedium_Test_medium, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ymedium_Test_medium = torch.tensor(ymedium_Test_medium, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xmedium_Test_deep = torch.tensor(Xmedium_Test_deep, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ymedium_Test_deep = torch.tensor(ymedium_Test_deep, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xmedium_Train = torch.nan_to_num(Xmedium_Train, nan=0.0)
   #yshallow_Train = torch.nan_to_num(yshallow_Train, nan=0.0)
   Xmedium_Test_shallow = torch.nan_to_num(Xmedium_Test_shallow, nan=0.0)
   Xmedium_Test_medium = torch.nan_to_num(Xmedium_Test_medium, nan=0.0)
   Xmedium_Test_deep = torch.nan_to_num(Xmedium_Test_deep, nan=0.0)
   #yshallow_Test = torch.nan_to_num(yshallow_Test, nan=0.0)

   # tensorizing the data for deep
   # ----------------------------------------
   Xdeep_Train = np.array(Xdeep_Train)
   ydeep_Train = np.array(ydeep_Train)
   Xdeep_Test_shallow = np.array(Xdeep_Test_shallow)
   ydeep_Test_shallow = np.array(ydeep_Test_shallow)
   Xdeep_Test_medium = np.array(Xdeep_Test_medium)
   ydeep_Test_medium = np.array(ydeep_Test_medium)
   Xdeep_Test_deep = np.array(Xdeep_Test_deep)
   ydeep_Test_deep = np.array(ydeep_Test_deep)

   Xdeep_Train = torch.tensor(Xdeep_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ydeep_Train = torch.tensor(ydeep_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xdeep_Test_shallow = torch.tensor(Xdeep_Test_shallow, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydeep_Test_shallow = torch.tensor(ydeep_Test_shallow, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xdeep_Test_medium = torch.tensor(Xdeep_Test_medium, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydeep_Test_medium = torch.tensor(ydeep_Test_medium, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xdeep_Test_deep = torch.tensor(Xdeep_Test_deep, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydeep_Test_deep = torch.tensor(ydeep_Test_deep, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xdeep_Train = torch.nan_to_num(Xdeep_Train, nan=0.0)
   #yshallow_Train = torch.nan_to_num(yshallow_Train, nan=0.0)
   Xdeep_Test_shallow = torch.nan_to_num(Xdeep_Test_shallow, nan=0.0)
   Xdeep_Test_medium = torch.nan_to_num(Xdeep_Test_medium, nan=0.0)
   Xdeep_Test_deep = torch.nan_to_num(Xdeep_Test_deep, nan=0.0)
   #yshallow_Test = torch.nan_to_num(yshallow_Test, nan=0.0)

   # ----------------------------------------
   # ----------------------------------------
   # ----------------------------------------

   # Reshape your input to add sequence length dimension for shallow
   Xshallow_Train = Xshallow_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xshallow_Test_shallow = Xshallow_Test_shallow.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xshallow_Test_medium = Xshallow_Test_medium.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xshallow_Test_deep = Xshallow_Test_deep.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_shallow = TensorDataset(Xshallow_Train, yshallow_Train)
   test_dataset_shallow_shallow = TensorDataset(Xshallow_Test_shallow, yshallow_Test_shallow)
   test_dataset_shallow_medium = TensorDataset(Xshallow_Test_medium, yshallow_Test_medium)
   test_dataset_shallow_deep = TensorDataset(Xshallow_Test_deep, yshallow_Test_deep)

   # Reshape your input to add sequence length dimension for medium
   Xmedium_Train = Xmedium_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xmedium_Test_shallow = Xmedium_Test_shallow.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xmedium_Test_medium = Xmedium_Test_medium.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xmedium_Test_deep = Xmedium_Test_deep.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_medium = TensorDataset(Xmedium_Train, ymedium_Train)
   test_dataset_medium_shallow = TensorDataset(Xmedium_Test_shallow, ymedium_Test_shallow)
   test_dataset_medium_medium = TensorDataset(Xmedium_Test_medium, ymedium_Test_medium)
   test_dataset_medium_deep = TensorDataset(Xmedium_Test_deep, ymedium_Test_deep)


   # Reshape your input to add sequence length dimension for deep
   Xdeep_Train = Xdeep_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdeep_Test_shallow = Xdeep_Test_shallow.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdeep_Test_medium = Xdeep_Test_medium.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdeep_Test_deep = Xdeep_Test_deep.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_deep = TensorDataset(Xdeep_Train, ydeep_Train)
   test_dataset_deep_shallow = TensorDataset(Xdeep_Test_shallow, ydeep_Test_shallow)
   test_dataset_deep_medium = TensorDataset(Xdeep_Test_medium, ydeep_Test_medium)
   test_dataset_deep_deep = TensorDataset(Xdeep_Test_deep, ydeep_Test_deep)


   # Create DataLoaders for train and test sets shallow
   train_loader_shallow = DataLoader(train_dataset_shallow, batch_size=batch_size, shuffle=True)
   test_loader_shallow_shallow = DataLoader(test_dataset_shallow_shallow, batch_size=batch_size, shuffle=False)
   test_loader_shallow_medium = DataLoader(test_dataset_shallow_medium, batch_size=batch_size, shuffle=False)
   test_loader_shallow_deep = DataLoader(test_dataset_shallow_deep, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets medium
   train_loader_medium = DataLoader(train_dataset_medium, batch_size=batch_size, shuffle=True)
   test_loader_medium_shallow = DataLoader(test_dataset_medium_shallow, batch_size=batch_size, shuffle=False)
   test_loader_medium_medium = DataLoader(test_dataset_medium_medium, batch_size=batch_size, shuffle=False)
   test_loader_medium_deep = DataLoader(test_dataset_medium_deep, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets deep
   train_loader_deep = DataLoader(train_dataset_deep, batch_size=batch_size, shuffle=True)
   test_loader_deep_shallow = DataLoader(test_dataset_deep_shallow, batch_size=batch_size, shuffle=False)
   test_loader_deep_medium = DataLoader(test_dataset_deep_medium, batch_size=batch_size, shuffle=False)
   test_loader_deep_deep = DataLoader(test_dataset_deep_deep, batch_size=batch_size, shuffle=False)

   # Define 3 IDS models
   model_shallow = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_medium = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_deep = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

   
   # Train and Test SHALLOW
   pr.prGreen("Start Training!")
   model_shallow.model_train(epochs = epochs,train_loader = train_loader_shallow)
   pr.prGreen("__ SHALLOW Trained!")
   model_shallow.check_model_nans(test_loader_shallow_shallow)
   model_shallow.check_model_nans(test_loader_shallow_medium)
   model_shallow.check_model_nans(test_loader_shallow_deep)
   roc_auc_shallow_shallow.append(round(model_shallow.evaluate_model_ROCAUC(test_loader_shallow_shallow),3))
   roc_auc_shallow_medium.append(round(model_shallow.evaluate_model_ROCAUC(test_loader_shallow_medium),3))
   roc_auc_shallow_deep.append(round(model_shallow.evaluate_model_ROCAUC(test_loader_shallow_deep),3))

   



   # Train and Test MEDIUM
   pr.prGreen("Start Training!")
   model_medium.model_train(epochs = epochs,train_loader = train_loader_medium)
   pr.prGreen("__ MEDIUM Trained!")
   model_medium.check_model_nans(test_loader_medium_shallow)
   model_medium.check_model_nans(test_loader_medium_medium)
   model_medium.check_model_nans(test_loader_medium_deep)
   roc_auc_medium_shallow.append(round(model_medium.evaluate_model_ROCAUC(test_loader_medium_shallow),3))
   roc_auc_medium_medium.append(round(model_medium.evaluate_model_ROCAUC(test_loader_medium_medium),3))
   roc_auc_medium_deep.append(round(model_medium.evaluate_model_ROCAUC(test_loader_medium_deep),3))

   


   
   # Train and Test DEEP
   pr.prGreen("Start Training!")
   model_deep.model_train(epochs = epochs,train_loader = train_loader_deep)
   pr.prGreen("__ DEEP Trained!")
   model_deep.check_model_nans(test_loader_deep_shallow)
   model_deep.check_model_nans(test_loader_deep_medium)
   model_deep.check_model_nans(test_loader_deep_deep)
   roc_auc_deep_shallow.append(round(model_deep.evaluate_model_ROCAUC(test_loader_deep_shallow),3))
   roc_auc_deep_medium.append(round(model_deep.evaluate_model_ROCAUC(test_loader_deep_medium),3))
   roc_auc_deep_deep.append(round(model_deep.evaluate_model_ROCAUC(test_loader_deep_deep),3)) 

   


# save data
with open(add + '/roc_auc_shallow_shallow.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_shallow_shallow)

with open(add + '/roc_auc_shallow_medium.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_shallow_medium)

with open(add + '/roc_auc_shallow_deep.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_shallow_deep)



with open(add + '/roc_auc_medium_shallow.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_medium_shallow)

with open(add + '/roc_auc_medium_medium.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_medium_medium)

with open(add + '/roc_auc_medium_deep.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_medium_deep)



with open(add + '/roc_auc_deep_shallow.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_deep_shallow)

with open(add + '/roc_auc_deep_medium.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_deep_medium)

with open(add + '/roc_auc_deep_deep.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_deep_deep)
