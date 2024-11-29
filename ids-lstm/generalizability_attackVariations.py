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
In this scenario we have 3 IDS models, trained with the different attack variations, base, on/off, gradual changing.
We test the generalizability of each model to detect the other attack types.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Generalizability - Attack Variations: ")

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


roc_auc_base_base = []           # records ROC-AUC of a model that is trained with the base attack variation data and tested with the base attack variation data
roc_auc_base_oo = []             # records ROC-AUC of a model that is trained with the base attack variation data and tested with the on/off attack variation data
roc_auc_base_dec = []            # records ROC-AUC of a model that is trained with the base attack variation data and tested with the gradual changing attack variation data
roc_auc_oo_base = []             # ...
roc_auc_oo_oo = []
roc_auc_oo_dec = []
roc_auc_dec_base = []
roc_auc_dec_oo = []
roc_auc_dec_dec = []





for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
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
   print("Normalize BASE: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
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
   

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   print(".................................")
   print("Normalize On/Off: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
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
   

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".................................")
   print("Normalize Gradual Changing: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
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
   Xbase_Test_base = torch.nan_to_num(Xbase_Test_base, nan=0.0)
   Xbase_Test_oo = torch.nan_to_num(Xbase_Test_oo, nan=0.0)
   Xbase_Test_dec = torch.nan_to_num(Xbase_Test_dec, nan=0.0)

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
   Xoo_Test_base = torch.nan_to_num(Xoo_Test_base, nan=0.0)
   Xoo_Test_oo = torch.nan_to_num(Xoo_Test_oo, nan=0.0)
   Xoo_Test_dec = torch.nan_to_num(Xoo_Test_dec, nan=0.0)

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
   Xdec_Test_base = torch.nan_to_num(Xdec_Test_base, nan=0.0)
   Xdec_Test_oo = torch.nan_to_num(Xdec_Test_oo, nan=0.0)
   Xdec_Test_dec = torch.nan_to_num(Xdec_Test_dec, nan=0.0)

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

   # Create DataLoaders for train and test sets gradual changing
   train_loader_dec = DataLoader(train_dataset_dec, batch_size=batch_size, shuffle=True)
   test_loader_dec_base = DataLoader(test_dataset_dec_base, batch_size=batch_size, shuffle=False)
   test_loader_dec_oo = DataLoader(test_dataset_dec_oo, batch_size=batch_size, shuffle=False)
   test_loader_dec_dec = DataLoader(test_dataset_dec_dec, batch_size=batch_size, shuffle=False)


   # Define 3 IDS models
   model_base = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_oo = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_dec = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

   
   # Train and Test base
   pr.prGreen("Start Training!")
   model_base.model_train(epochs = epochs,train_loader = train_loader_base)
   pr.prGreen("__ BASE Trained!")
   model_base.check_model_nans(test_loader_base_base)
   model_base.check_model_nans(test_loader_base_oo)
   model_base.check_model_nans(test_loader_base_dec)
   roc_auc_base_base.append(round(model_base.evaluate_model_ROCAUC(test_loader_base_base),3))
   roc_auc_base_oo.append(round(model_base.evaluate_model_ROCAUC(test_loader_base_oo),3))
   roc_auc_base_dec.append(round(model_base.evaluate_model_ROCAUC(test_loader_base_dec),3))

   



   # Train and Test OO
   pr.prGreen("Start Training!")
   model_oo.model_train(epochs = epochs,train_loader = train_loader_oo)
   pr.prGreen("__ OO Trained!")
   model_oo.check_model_nans(test_loader_oo_base)
   model_oo.check_model_nans(test_loader_oo_oo)
   model_oo.check_model_nans(test_loader_oo_dec)
   roc_auc_oo_base.append(round(model_oo.evaluate_model_ROCAUC(test_loader_oo_base),3))
   roc_auc_oo_oo.append(round(model_oo.evaluate_model_ROCAUC(test_loader_oo_oo),3))
   roc_auc_oo_dec.append(round(model_oo.evaluate_model_ROCAUC(test_loader_oo_dec),3))

   


   
   # Train and Test Grad.
   pr.prGreen("Start Training!")
   model_dec.model_train(epochs = epochs,train_loader = train_loader_dec)
   pr.prGreen("__ DEC Trained!")
   model_dec.check_model_nans(test_loader_dec_base)
   model_dec.check_model_nans(test_loader_dec_oo)
   model_dec.check_model_nans(test_loader_dec_dec)
   roc_auc_dec_base.append(round(model_dec.evaluate_model_ROCAUC(test_loader_dec_base),3))
   roc_auc_dec_oo.append(round(model_dec.evaluate_model_ROCAUC(test_loader_dec_oo),3))
   roc_auc_dec_dec.append(round(model_dec.evaluate_model_ROCAUC(test_loader_dec_dec),3)) 

   


# save data
with open(add + '/roc_auc_base_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_base_base)

with open(add + '/roc_auc_base_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_base_oo)

with open(add + '/roc_auc_base_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_base_dec)



with open(add + '/roc_auc_oo_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_oo_base)

with open(add + '/roc_auc_oo_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_oo_oo)

with open(add + '/roc_auc_oo_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_oo_dec)



with open(add + '/roc_auc_dec_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_dec_base)

with open(add + '/roc_auc_dec_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_dec_oo)

with open(add + '/roc_auc_dec_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_dec_dec)


























