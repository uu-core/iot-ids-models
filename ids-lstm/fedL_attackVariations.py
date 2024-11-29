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
In this scenario we have 1 IDS models, trained with horizontal federated learning method. Three local models (clients) 
contribute in training the global model: the model that is trained with the base, the on/off and the gradual changing attack variations.
We test the generalizability of the global model to detect the base, the on/off and the gradual changing data.
We train and test the model --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Global Model with the federated learning method - test on the attack variations (base, on/off and gradual changing)")

# Add the arguments
parser.add_argument('-lr', type=float, default = 0.001, help = 'Learning rate')
parser.add_argument('-run', type= int, default = 1, help = 'Number of Runs')
parser.add_argument('-round', type= int, default = 1, help = 'Number of Rounds')
parser.add_argument('-batch',type = int, default = 2048, help = 'Batch Size')
parser.add_argument('-epc',type = int, default = 1, help = 'Epochs Per Client')
parser.add_argument('-add',type = str, default = '/Path/to/...', help = 'Save Address')



# Parse the arguments
args = parser.parse_args()

# Read the arguments
lr = args.lr                     # learninng rate in gradient descent
runs = args.run                  # number of runs, to generate a boxplot we trained and tested each model 10 times
rnds = args.round                # the number of rounds that the global model gets trained
batch_size = args.batch          # batch size
add = args.add                   # address to save the results
epochs_per_client =args.epc      # the number of epochs that each client gets trained in each round

#Defining global variables
hidden_dim = 10                  # LSTM hyperparameter
fc_hidden_dim = 10               # LSTM hyperparameter
num_layers = 1                   # LSTM hyperparameter
output_dim = 2                   # output: attack, non-attack
sequence_length = 10             # sequence length (10 min)
dropout_rate = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # if gpu is available to use or not






roc_auc_fed_base_base = []    # records ROC-AUC of a model that is trained the FL method and tested with the base data (normalized with the client's min and max with the base data)
roc_auc_fed_base_oo = []      # records ROC-AUC of a model that is trained the FL method and tested with the oo data (normalized with the client's min and max with the base data)
roc_auc_fed_base_dec = []     # records ROC-AUC of a model that is trained the FL method and tested with the gradual changing data (normalized with the client's min and max with the base data)

roc_auc_fed_oo_base = []      # ...
roc_auc_fed_oo_oo = []
roc_auc_fed_oo_dec = []

roc_auc_fed_dec_base = []
roc_auc_fed_dec_oo = []
roc_auc_fed_dec_dec = []




for run in range(runs):
   pr.prGreen("run " + str(run))
   print(".................................")
   #fed_Train, fed_Test = aggregate.aggregate_list_fed()
   base_Train, base_Test = aggregate.aggregate_list_base()
   oo_Train, oo_Test = aggregate.aggregate_list_oo()
   dec_Train, dec_Test = aggregate.aggregate_list_dec()

   
   pr.prGreen(len(base_Train))
   pr.prGreen(len(base_Test))
   pr.prRed(len(oo_Train))
   pr.prRed(len(oo_Test))
   pr.prGreen(len(dec_Train))
   pr.prGreen(len(dec_Test))
   

   

   print(".................................")
   print("Normalize base: ")
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in base_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   base_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   base_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   base_max = base_all_maxs_df.max(axis = 0)
   base_min = base_all_mins_df.min(axis = 0)
   base_normalized_dfs_train = [df.apply(lambda x: (x - base_min[x.name]) / (base_max[x.name] - base_min[x.name])) for df in base_Train]
   #base_normalized_dfs_test = [df.apply(lambda x: (x - base_min[x.name]) / (base_max[x.name] - base_min[x.name])) for df in _base_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   print(".................................")
   print("Normalize OO : ")
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in oo_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   oo_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   oo_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   oo_max = oo_all_maxs_df.max(axis = 0)
   oo_min = oo_all_mins_df.min(axis = 0)
   oo_normalized_dfs_train = [df.apply(lambda x: (x - oo_min[x.name]) / (oo_max[x.name] - oo_min[x.name])) for df in oo_Train]
   #oo_normalized_dfs_test = [df.apply(lambda x: (x - oo_min[x.name]) / (oo_max[x.name] - oo_min[x.name])) for df in _oo_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".................................")
   print("Normalize DEC : ")
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in dec_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   dec_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   dec_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   dec_max = dec_all_maxs_df.max(axis = 0)
   dec_min = dec_all_mins_df.min(axis = 0)
   dec_normalized_dfs_train = [df.apply(lambda x: (x - dec_min[x.name]) / (dec_max[x.name] - dec_min[x.name])) for df in dec_Train]
   #dec_normalized_dfs_test = [df.apply(lambda x: (x - dec_min[x.name]) / (dec_max[x.name] - dec_min[x.name])) for df in _dec_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)



   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   base_normalized_dfs_test_base = [df.apply(lambda x: (x - base_min[x.name]) / (base_max[x.name] - base_min[x.name])) for df in base_Test]
   base_normalized_dfs_test_oo = [df.apply(lambda x: (x - base_min[x.name]) / (base_max[x.name] - base_min[x.name])) for df in oo_Test]
   base_normalized_dfs_test_dec = [df.apply(lambda x: (x - base_min[x.name]) / (base_max[x.name] - base_min[x.name])) for df in dec_Test]

   oo_normalized_dfs_test_base = [df.apply(lambda x: (x - oo_min[x.name]) / (oo_max[x.name] - oo_min[x.name])) for df in base_Test]
   oo_normalized_dfs_test_oo = [df.apply(lambda x: (x - oo_min[x.name]) / (oo_max[x.name] - oo_min[x.name])) for df in oo_Test]
   oo_normalized_dfs_test_dec = [df.apply(lambda x: (x - oo_min[x.name]) / (oo_max[x.name] - oo_min[x.name])) for df in dec_Test]

   dec_normalized_dfs_test_base = [df.apply(lambda x: (x - dec_min[x.name]) / (dec_max[x.name] - dec_min[x.name])) for df in base_Test]
   dec_normalized_dfs_test_oo = [df.apply(lambda x: (x - dec_min[x.name]) / (dec_max[x.name] - dec_min[x.name])) for df in oo_Test]
   dec_normalized_dfs_test_dec = [df.apply(lambda x: (x - dec_min[x.name]) / (dec_max[x.name] - dec_min[x.name])) for df in dec_Test]
   # ------------------------------------
   # make them all seq and the concatenate
   sequencedbase_Train = [seqMaker.seq_maker(df,10) for df in base_normalized_dfs_train]
   sequencedbase_Test_base = [seqMaker.seq_maker(df,10) for df in base_normalized_dfs_test_base]
   sequencedbase_Test_oo = [seqMaker.seq_maker(df,10) for df in base_normalized_dfs_test_oo]
   sequencedbase_Test_dec = [seqMaker.seq_maker(df,10) for df in base_normalized_dfs_test_dec]

   sequencedoo_Train = [seqMaker.seq_maker(df,10) for df in oo_normalized_dfs_train]
   sequencedoo_Test_base = [seqMaker.seq_maker(df,10) for df in oo_normalized_dfs_test_base]
   sequencedoo_Test_oo = [seqMaker.seq_maker(df,10) for df in oo_normalized_dfs_test_oo]
   sequencedoo_Test_dec = [seqMaker.seq_maker(df,10) for df in oo_normalized_dfs_test_dec]

   sequenceddec_Train = [seqMaker.seq_maker(df,10) for df in dec_normalized_dfs_train]
   sequenceddec_Test_base = [seqMaker.seq_maker(df,10) for df in dec_normalized_dfs_test_base]
   sequenceddec_Test_oo = [seqMaker.seq_maker(df,10) for df in dec_normalized_dfs_test_oo]
   sequenceddec_Test_dec = [seqMaker.seq_maker(df,10) for df in dec_normalized_dfs_test_dec]


   sequencedbase_Train = pd.concat(sequencedbase_Train, ignore_index=True)
   sequencedbase_Test_base = pd.concat(sequencedbase_Test_base, ignore_index=True)
   sequencedbase_Test_oo = pd.concat(sequencedbase_Test_oo, ignore_index=True)
   sequencedbase_Test_dec = pd.concat(sequencedbase_Test_dec, ignore_index=True)

   sequencedoo_Train = pd.concat(sequencedoo_Train, ignore_index=True)
   sequencedoo_Test_base = pd.concat(sequencedoo_Test_base, ignore_index=True)
   sequencedoo_Test_oo = pd.concat(sequencedoo_Test_oo, ignore_index=True)
   sequencedoo_Test_dec = pd.concat(sequencedoo_Test_dec, ignore_index=True)

   sequenceddec_Train = pd.concat(sequenceddec_Train, ignore_index=True)
   sequenceddec_Test_base = pd.concat(sequenceddec_Test_base, ignore_index=True)
   sequenceddec_Test_oo = pd.concat(sequenceddec_Test_oo, ignore_index=True)
   sequenceddec_Test_dec = pd.concat(sequenceddec_Test_dec, ignore_index=True)


   # extract X and y for base, oo, dec
   Xbase_Train = sequencedbase_Train.iloc[:, :-1].values  # All columns except the last one
   ybase_Train = sequencedbase_Train.iloc[:, -1].values  # The last column
   Xoo_Train = sequencedoo_Train.iloc[:, :-1].values  # All columns except the last one
   yoo_Train = sequencedoo_Train.iloc[:, -1].values  # The last column
   Xdec_Train = sequenceddec_Train.iloc[:, :-1].values  # All columns except the last one
   ydec_Train = sequenceddec_Train.iloc[:, -1].values  # The last column

   Xbase_Test_base = sequencedbase_Test_base.iloc[:, :-1].values  # All columns except the last one
   ybase_Test_base = sequencedbase_Test_base.iloc[:, -1].values  # The last column
   Xbase_Test_oo = sequencedbase_Test_oo.iloc[:, :-1].values  # All columns except the last one
   ybase_Test_oo = sequencedbase_Test_oo.iloc[:, -1].values  # The last column
   Xbase_Test_dec = sequencedbase_Test_dec.iloc[:, :-1].values  # All columns except the last one
   ybase_Test_dec = sequencedbase_Test_dec.iloc[:, -1].values  # The last column

   Xoo_Test_base = sequencedoo_Test_base.iloc[:, :-1].values  # All columns except the last one
   yoo_Test_base = sequencedoo_Test_base.iloc[:, -1].values  # The last column
   Xoo_Test_oo = sequencedoo_Test_oo.iloc[:, :-1].values  # All columns except the last one
   yoo_Test_oo = sequencedoo_Test_oo.iloc[:, -1].values  # The last column
   Xoo_Test_dec = sequencedoo_Test_dec.iloc[:, :-1].values  # All columns except the last one
   yoo_Test_dec = sequencedoo_Test_dec.iloc[:, -1].values  # The last column

   Xdec_Test_base = sequenceddec_Test_base.iloc[:, :-1].values  # All columns except the last one
   ydec_Test_base = sequenceddec_Test_base.iloc[:, -1].values  # The last column
   Xdec_Test_oo = sequenceddec_Test_oo.iloc[:, :-1].values  # All columns except the last one
   ydec_Test_oo = sequenceddec_Test_oo.iloc[:, -1].values  # The last column
   Xdec_Test_dec = sequenceddec_Test_dec.iloc[:, :-1].values  # All columns except the last one
   ydec_Test_dec = sequenceddec_Test_dec.iloc[:, -1].values  # The last column



   # tensorizing the data for base
   # ----------------------------------------
   Xbase_Train = np.array(Xbase_Train)
   ybase_Train = np.array(ybase_Train)
   Xbase_Test_base = np.array(Xbase_Test_base)
   ybase_Test_base = np.array(ybase_Test_base)
   Xbase_Test_oo = np.array(Xbase_Test_oo)
   ybase_Test_oo = np.array(ybase_Test_oo)
   Xbase_Test_dec = np.array(Xbase_Test_dec)
   ybase_Test_dec = np.array(ybase_Test_dec)

   Xbase_Train = torch.tensor(Xbase_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ybase_Train = torch.tensor(ybase_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xbase_Test_base = torch.tensor(Xbase_Test_base, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ybase_Test_base = torch.tensor(ybase_Test_base, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xbase_Test_oo = torch.tensor(Xbase_Test_oo, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ybase_Test_oo = torch.tensor(ybase_Test_oo, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xbase_Test_dec = torch.tensor(Xbase_Test_dec, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ybase_Test_dec = torch.tensor(ybase_Test_dec, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xbase_Train = torch.nan_to_num(Xbase_Train, nan=0.0)
   #ybase_Train = torch.nan_to_num(ybase_Train, nan=0.0)
   Xbase_Test_base = torch.nan_to_num(Xbase_Test_base, nan=0.0)
   Xbase_Test_oo = torch.nan_to_num(Xbase_Test_oo, nan=0.0)
   Xbase_Test_dec = torch.nan_to_num(Xbase_Test_dec, nan=0.0)
   #ybase_Test = torch.nan_to_num(ybase_Test, nan=0.0)

   # tensorizing the data for oo
   # ----------------------------------------
   Xoo_Train = np.array(Xoo_Train)
   yoo_Train = np.array(yoo_Train)
   Xoo_Test_base = np.array(Xoo_Test_base)
   yoo_Test_base = np.array(yoo_Test_base)
   Xoo_Test_oo = np.array(Xoo_Test_oo)
   yoo_Test_oo = np.array(yoo_Test_oo)
   Xoo_Test_dec = np.array(Xoo_Test_dec)
   yoo_Test_dec = np.array(yoo_Test_dec)

   Xoo_Train = torch.tensor(Xoo_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   yoo_Train = torch.tensor(yoo_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xoo_Test_base = torch.tensor(Xoo_Test_base, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yoo_Test_base = torch.tensor(yoo_Test_base, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xoo_Test_oo = torch.tensor(Xoo_Test_oo, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yoo_Test_oo = torch.tensor(yoo_Test_oo, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xoo_Test_dec = torch.tensor(Xoo_Test_dec, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yoo_Test_dec = torch.tensor(yoo_Test_dec, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xoo_Train = torch.nan_to_num(Xoo_Train, nan=0.0)
   #ybase_Train = torch.nan_to_num(ybase_Train, nan=0.0)
   Xoo_Test_base = torch.nan_to_num(Xoo_Test_base, nan=0.0)
   Xoo_Test_oo = torch.nan_to_num(Xoo_Test_oo, nan=0.0)
   Xoo_Test_dec = torch.nan_to_num(Xoo_Test_dec, nan=0.0)
   #ybase_Test = torch.nan_to_num(ybase_Test, nan=0.0)

   # tensorizing the data for dec
   # ----------------------------------------
   Xdec_Train = np.array(Xdec_Train)
   ydec_Train = np.array(ydec_Train)
   Xdec_Test_base = np.array(Xdec_Test_base)
   ydec_Test_base = np.array(ydec_Test_base)
   Xdec_Test_oo = np.array(Xdec_Test_oo)
   ydec_Test_oo = np.array(ydec_Test_oo)
   Xdec_Test_dec = np.array(Xdec_Test_dec)
   ydec_Test_dec = np.array(ydec_Test_dec)

   Xdec_Train = torch.tensor(Xdec_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ydec_Train = torch.tensor(ydec_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xdec_Test_base = torch.tensor(Xdec_Test_base, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydec_Test_base = torch.tensor(ydec_Test_base, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xdec_Test_oo = torch.tensor(Xdec_Test_oo, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydec_Test_oo = torch.tensor(ydec_Test_oo, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xdec_Test_dec = torch.tensor(Xdec_Test_dec, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydec_Test_dec = torch.tensor(ydec_Test_dec, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xdec_Train = torch.nan_to_num(Xdec_Train, nan=0.0)
   #ybase_Train = torch.nan_to_num(ybase_Train, nan=0.0)
   Xdec_Test_base = torch.nan_to_num(Xdec_Test_base, nan=0.0)
   Xdec_Test_oo = torch.nan_to_num(Xdec_Test_oo, nan=0.0)
   Xdec_Test_dec = torch.nan_to_num(Xdec_Test_dec, nan=0.0)
   #ybase_Test = torch.nan_to_num(ybase_Test, nan=0.0)

   # ----------------------------------------
   # ----------------------------------------
   # ----------------------------------------


   # Reshape your input to add sequence length dimension for base
   Xbase_Train = Xbase_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xbase_Test_base = Xbase_Test_base.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xbase_Test_oo = Xbase_Test_oo.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xbase_Test_dec = Xbase_Test_dec.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_base = TensorDataset(Xbase_Train, ybase_Train)
   test_dataset_base_base = TensorDataset(Xbase_Test_base, ybase_Test_base)
   test_dataset_base_oo = TensorDataset(Xbase_Test_oo, ybase_Test_oo)
   test_dataset_base_dec = TensorDataset(Xbase_Test_dec, ybase_Test_dec)

   # Reshape your input to add sequence length dimension for oo
   Xoo_Train = Xoo_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xoo_Test_base = Xoo_Test_base.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xoo_Test_oo = Xoo_Test_oo.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xoo_Test_dec = Xoo_Test_dec.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_oo = TensorDataset(Xoo_Train, yoo_Train)
   test_dataset_oo_base = TensorDataset(Xoo_Test_base, yoo_Test_base)
   test_dataset_oo_oo = TensorDataset(Xoo_Test_oo, yoo_Test_oo)
   test_dataset_oo_dec = TensorDataset(Xoo_Test_dec, yoo_Test_dec)


   # Reshape your input to add sequence length dimension for dec
   Xdec_Train = Xdec_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdec_Test_base = Xdec_Test_base.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdec_Test_oo = Xdec_Test_oo.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdec_Test_dec = Xdec_Test_dec.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_dec = TensorDataset(Xdec_Train, ydec_Train)
   test_dataset_dec_base = TensorDataset(Xdec_Test_base, ydec_Test_base)
   test_dataset_dec_oo = TensorDataset(Xdec_Test_oo, ydec_Test_oo)
   test_dataset_dec_dec = TensorDataset(Xdec_Test_dec, ydec_Test_dec)


   # Create DataLoaders for train and test sets base
   train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
   test_loader_base_base = DataLoader(test_dataset_base_base, batch_size=batch_size, shuffle=False)
   test_loader_base_oo = DataLoader(test_dataset_base_oo, batch_size=batch_size, shuffle=False)
   test_loader_base_dec = DataLoader(test_dataset_base_dec, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets oo
   train_loader_oo = DataLoader(train_dataset_oo, batch_size=batch_size, shuffle=True)
   test_loader_oo_base = DataLoader(test_dataset_oo_base, batch_size=batch_size, shuffle=False)
   test_loader_oo_oo = DataLoader(test_dataset_oo_oo, batch_size=batch_size, shuffle=False)
   test_loader_oo_dec = DataLoader(test_dataset_oo_dec, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets dec
   train_loader_dec = DataLoader(train_dataset_dec, batch_size=batch_size, shuffle=True)
   test_loader_dec_base = DataLoader(test_dataset_dec_base, batch_size=batch_size, shuffle=False)
   test_loader_dec_oo = DataLoader(test_dataset_dec_oo, batch_size=batch_size, shuffle=False)
   test_loader_dec_dec = DataLoader(test_dataset_dec_dec, batch_size=batch_size, shuffle=False)



   #############################################################
   #############################################################
   ##### setup clients

   # Initialize client A and client B's models
   client_base = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_oo = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_dec = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   
                      
   clients = [client_base, client_oo, client_dec]
   training_loaders = [train_loader_base, train_loader_oo, train_loader_dec]


   server_model = LSTM_FED.federated_training(clients[0], clients, global_epochs = rnds, client_epochs = epochs_per_client, training_loaders = training_loaders)


   roc_auc_fed_base_base.append(round(server_model.evaluate_model_ROCAUC(test_loader_base_base),3))
   roc_auc_fed_base_oo.append(round(server_model.evaluate_model_ROCAUC(test_loader_base_oo),3))
   roc_auc_fed_base_dec.append(round(server_model.evaluate_model_ROCAUC(test_loader_base_dec),3))

   roc_auc_fed_oo_base.append(round(server_model.evaluate_model_ROCAUC(test_loader_oo_base),3))
   roc_auc_fed_oo_oo.append(round(server_model.evaluate_model_ROCAUC(test_loader_oo_oo),3))
   roc_auc_fed_oo_dec.append(round(server_model.evaluate_model_ROCAUC(test_loader_oo_dec),3))

   roc_auc_fed_dec_base.append(round(server_model.evaluate_model_ROCAUC(test_loader_dec_base),3))
   roc_auc_fed_dec_oo.append(round(server_model.evaluate_model_ROCAUC(test_loader_dec_oo),3))
   roc_auc_fed_dec_dec.append(round(server_model.evaluate_model_ROCAUC(test_loader_dec_dec),3))





print(roc_auc_fed_base_base)
print(roc_auc_fed_base_oo)
print(roc_auc_fed_base_dec)
print(roc_auc_fed_oo_base)
print(roc_auc_fed_oo_oo)
print(roc_auc_fed_oo_dec)
print(roc_auc_fed_dec_base)
print(roc_auc_fed_dec_oo)
print(roc_auc_fed_dec_dec)


# save results
with open(add + '/roc_auc_fed_base_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_base_base)

with open(add + '/roc_auc_fed_base_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_base_oo)

with open(add + '/roc_auc_fed_base_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_base_dec)


with open(add + '/roc_auc_fed_oo_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_oo_base)

with open(add + '/roc_auc_fed_oo_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_oo_oo)

with open(add + '/roc_auc_fed_oo_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_oo_dec)


with open(add + '/roc_auc_fed_dec_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_dec_base)

with open(add + '/roc_auc_fed_dec_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_dec_oo)

with open(add + '/roc_auc_fed_dec_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_fed_dec_dec)

