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
In this scenario we have 4 IDS models, trained in a network with 5, 10, 15 and 20 nodes.
We test the generalizability of each model in other networks with different sizes.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Generalizability - Network Size: ")

# Add the arguments
parser.add_argument('-lr', type=float, default = 0.001, help = 'Learning rate')
parser.add_argument('-ep', type= int, default = 1, help = 'Number of Epochs')
parser.add_argument('-run', type= int, default = 1, help = 'Number of Runs')
parser.add_argument('-batch',type = int, default = 2048, help = 'Batch Size')
parser.add_argument('-add',type = str, default = '/Path/to/...', help = 'Save Address')

# Parse the arguments
args = parser.parse_args()

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


roc_auc_5_5 = []                 # records ROC-AUC of a model that is trained in the network with 5 nodes and tested in a network with 5 nodes
roc_auc_5_10 = []                # records ROC-AUC of a model that is trained in the network with 5 nodes and tested in a network with 10 nodes
roc_auc_5_15 = []                # records ROC-AUC of a model that is trained in the network with 5 nodes and tested in a network with 15 nodes
roc_auc_5_20 = []                # records ROC-AUC of a model that is trained in the network with 5 nodes and tested in a network with 20 nodes
roc_auc_10_5 = []                # ... 
roc_auc_10_10 = []
roc_auc_10_15 = []
roc_auc_10_20 = []
roc_auc_15_5 = []
roc_auc_15_10 = []
roc_auc_15_15 = []
roc_auc_15_20 = []
roc_auc_20_5 = []
roc_auc_20_10 = []
roc_auc_20_15 = []
roc_auc_20_20 = []






   




for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   
   _5_Train, _5_Test = aggregate.aggregate_list_5()
   _10_Train, _10_Test = aggregate.aggregate_list_10()
   _15_Train, _15_Test = aggregate.aggregate_list_15()
   _20_Train, _20_Test = aggregate.aggregate_list_20()

   print(".................................")
   print("Normalize 5: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _5_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __5_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __5_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __5_max = __5_all_maxs_df.max(axis = 0)
   __5_min = __5_all_mins_df.min(axis = 0)
   __5_normalized_dfs_train = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _5_Train]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   
   print(".................................")
   print("Normalize 10 : ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _10_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __10_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __10_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __10_max = __10_all_maxs_df.max(axis = 0)
   __10_min = __10_all_mins_df.min(axis = 0)
   __10_normalized_dfs_train = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _10_Train]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".................................")
   print("Normalize 15 : ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _15_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __15_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __15_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __15_max = __15_all_maxs_df.max(axis = 0)
   __15_min = __15_all_mins_df.min(axis = 0)
   __15_normalized_dfs_train = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _15_Train]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   print(".................................")
   print("Normalize 20 : ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _20_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __20_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __20_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __20_max = __20_all_maxs_df.max(axis = 0)
   __20_min = __20_all_mins_df.min(axis = 0)
   __20_normalized_dfs_train = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _20_Train]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   __5_normalized_dfs_test_5 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _5_Test]    
   __5_normalized_dfs_test_10 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _10_Test]
   __5_normalized_dfs_test_15 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _15_Test]
   __5_normalized_dfs_test_20 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _20_Test]

   __10_normalized_dfs_test_5 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _5_Test]    
   __10_normalized_dfs_test_10 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _10_Test]
   __10_normalized_dfs_test_15 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _15_Test]
   __10_normalized_dfs_test_20 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _20_Test]

   __15_normalized_dfs_test_5 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _5_Test]    
   __15_normalized_dfs_test_10 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _10_Test]
   __15_normalized_dfs_test_15 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _15_Test]
   __15_normalized_dfs_test_20 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _20_Test]

   __20_normalized_dfs_test_5 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _5_Test]    
   __20_normalized_dfs_test_10 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _10_Test]
   __20_normalized_dfs_test_15 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _15_Test]
   __20_normalized_dfs_test_20 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _20_Test]



   # ------------------------------------
   # make them all seq and the concatenate
   sequenced_5_Train = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_train]
   sequenced_5_Test_5 = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_test_5]
   sequenced_5_Test_10 = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_test_10]
   sequenced_5_Test_15 = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_test_15]
   sequenced_5_Test_20 = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_test_20]

   sequenced_10_Train = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_train]
   sequenced_10_Test_5 = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_test_5]
   sequenced_10_Test_10 = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_test_10]
   sequenced_10_Test_15 = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_test_15]
   sequenced_10_Test_20 = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_test_20]

   sequenced_15_Train = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_train]
   sequenced_15_Test_5 = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_test_5]
   sequenced_15_Test_10 = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_test_10]
   sequenced_15_Test_15 = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_test_15]
   sequenced_15_Test_20 = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_test_20]

   sequenced_20_Train = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_train]
   sequenced_20_Test_5 = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_test_5]
   sequenced_20_Test_10 = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_test_10]
   sequenced_20_Test_15 = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_test_15]
   sequenced_20_Test_20 = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_test_20]

   # 

   sequenced_5_Train = pd.concat(sequenced_5_Train, ignore_index=True)
   sequenced_5_Test_5 = pd.concat(sequenced_5_Test_5, ignore_index=True)
   sequenced_5_Test_10 = pd.concat(sequenced_5_Test_10, ignore_index=True)
   sequenced_5_Test_15 = pd.concat(sequenced_5_Test_15, ignore_index=True)
   sequenced_5_Test_20 = pd.concat(sequenced_5_Test_20, ignore_index=True)

   sequenced_10_Train = pd.concat(sequenced_10_Train, ignore_index=True)
   sequenced_10_Test_5 = pd.concat(sequenced_10_Test_5, ignore_index=True)
   sequenced_10_Test_10 = pd.concat(sequenced_10_Test_10, ignore_index=True)
   sequenced_10_Test_15 = pd.concat(sequenced_10_Test_15, ignore_index=True)
   sequenced_10_Test_20 = pd.concat(sequenced_10_Test_20, ignore_index=True)

   sequenced_15_Train = pd.concat(sequenced_15_Train, ignore_index=True)
   sequenced_15_Test_5 = pd.concat(sequenced_15_Test_5, ignore_index=True)
   sequenced_15_Test_10 = pd.concat(sequenced_15_Test_10, ignore_index=True)
   sequenced_15_Test_15 = pd.concat(sequenced_15_Test_15, ignore_index=True)
   sequenced_15_Test_20 = pd.concat(sequenced_15_Test_20, ignore_index=True)

   sequenced_20_Train = pd.concat(sequenced_20_Train, ignore_index=True)
   sequenced_20_Test_5 = pd.concat(sequenced_20_Test_5, ignore_index=True)
   sequenced_20_Test_10 = pd.concat(sequenced_20_Test_10, ignore_index=True)
   sequenced_20_Test_15 = pd.concat(sequenced_20_Test_15, ignore_index=True)
   sequenced_20_Test_20 = pd.concat(sequenced_20_Test_20, ignore_index=True)

   
   
   
   # extract X and y for 5 to 20
   X_5_train = sequenced_5_Train.iloc[:, :-1].values  # All columns except the last one
   y_5_train = sequenced_5_Train.iloc[:, -1].values  # The last column
   X_10_train = sequenced_10_Train.iloc[:, :-1].values  # All columns except the last one
   y_10_train = sequenced_10_Train.iloc[:, -1].values  # The last column
   X_15_train = sequenced_15_Train.iloc[:, :-1].values  # All columns except the last one
   y_15_train = sequenced_15_Train.iloc[:, -1].values  # The last column
   X_20_train = sequenced_20_Train.iloc[:, :-1].values  # All columns except the last one
   y_20_train = sequenced_20_Train.iloc[:, -1].values  # The last column


   X_5_test_5 = sequenced_5_Test_5.iloc[:, :-1].values  # All columns except the last one
   y_5_test_5 = sequenced_5_Test_5.iloc[:, -1].values  # The last column
   X_5_test_10 = sequenced_5_Test_10.iloc[:, :-1].values  # All columns except the last one
   y_5_test_10 = sequenced_5_Test_10.iloc[:, -1].values  # The last column
   X_5_test_15 = sequenced_5_Test_15.iloc[:, :-1].values  # All columns except the last one
   y_5_test_15 = sequenced_5_Test_15.iloc[:, -1].values  # The last column
   X_5_test_20 = sequenced_5_Test_20.iloc[:, :-1].values  # All columns except the last one
   y_5_test_20 = sequenced_5_Test_20.iloc[:, -1].values  # The last column


   X_10_test_5 = sequenced_10_Test_5.iloc[:, :-1].values  # All columns except the last one
   y_10_test_5 = sequenced_10_Test_5.iloc[:, -1].values  # The last column
   X_10_test_10 = sequenced_10_Test_10.iloc[:, :-1].values  # All columns except the last one
   y_10_test_10 = sequenced_10_Test_10.iloc[:, -1].values  # The last column
   X_10_test_15 = sequenced_10_Test_15.iloc[:, :-1].values  # All columns except the last one
   y_10_test_15 = sequenced_10_Test_15.iloc[:, -1].values  # The last column
   X_10_test_20 = sequenced_10_Test_20.iloc[:, :-1].values  # All columns except the last one
   y_10_test_20 = sequenced_10_Test_20.iloc[:, -1].values  # The last column


   X_15_test_5 = sequenced_15_Test_5.iloc[:, :-1].values  # All columns except the last one
   y_15_test_5 = sequenced_15_Test_5.iloc[:, -1].values  # The last column
   X_15_test_10 = sequenced_15_Test_10.iloc[:, :-1].values  # All columns except the last one
   y_15_test_10 = sequenced_15_Test_10.iloc[:, -1].values  # The last column
   X_15_test_15 = sequenced_15_Test_15.iloc[:, :-1].values  # All columns except the last one
   y_15_test_15 = sequenced_15_Test_15.iloc[:, -1].values  # The last column
   X_15_test_20 = sequenced_15_Test_20.iloc[:, :-1].values  # All columns except the last one
   y_15_test_20 = sequenced_15_Test_20.iloc[:, -1].values  # The last column


   X_20_test_5 = sequenced_20_Test_5.iloc[:, :-1].values  # All columns except the last one
   y_20_test_5 = sequenced_20_Test_5.iloc[:, -1].values  # The last column
   X_20_test_10 = sequenced_20_Test_10.iloc[:, :-1].values  # All columns except the last one
   y_20_test_10 = sequenced_20_Test_10.iloc[:, -1].values  # The last column
   X_20_test_15 = sequenced_20_Test_15.iloc[:, :-1].values  # All columns except the last one
   y_20_test_15 = sequenced_20_Test_15.iloc[:, -1].values  # The last column
   X_20_test_20 = sequenced_20_Test_20.iloc[:, :-1].values  # All columns except the last one
   y_20_test_20 = sequenced_20_Test_20.iloc[:, -1].values  # The last column
   
   # ----------------------------------------------------------------

   # tensorizing the data for 5
   X_5_train = np.array(X_5_train)
   y_5_train = np.array(y_5_train)
   X_5_test_5 = np.array(X_5_test_5)
   y_5_test_5 = np.array(y_5_test_5)
   X_5_test_10 = np.array(X_5_test_10)
   y_5_test_10 = np.array(y_5_test_10)
   X_5_test_15 = np.array(X_5_test_15)
   y_5_test_15 = np.array(y_5_test_15)
   X_5_test_20 = np.array(X_5_test_20)
   y_5_test_20 = np.array(y_5_test_20)

   X_5_train = torch.tensor(X_5_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_5_train = torch.tensor(y_5_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   X_5_test_5 = torch.tensor(X_5_test_5, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_5_test_5 = torch.tensor(y_5_test_5, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_5_test_10 = torch.tensor(X_5_test_10, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_5_test_10 = torch.tensor(y_5_test_10, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_5_test_15 = torch.tensor(X_5_test_15, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_5_test_15 = torch.tensor(y_5_test_15, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_5_test_20 = torch.tensor(X_5_test_20, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_5_test_20 = torch.tensor(y_5_test_20, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_5_train = torch.nan_to_num(X_5_train, nan=0.0)
   X_5_test_5 = torch.nan_to_num(X_5_test_5, nan=0.0)
   X_5_test_10 = torch.nan_to_num(X_5_test_10, nan=0.0)
   X_5_test_15 = torch.nan_to_num(X_5_test_15, nan=0.0)
   X_5_test_20 = torch.nan_to_num(X_5_test_20, nan=0.0)


   # tensorizing the data for 10
   X_10_train = np.array(X_10_train)
   y_10_train = np.array(y_10_train)
   X_10_test_5 = np.array(X_10_test_5)
   y_10_test_5 = np.array(y_10_test_5)
   X_10_test_10 = np.array(X_10_test_10)
   y_10_test_10 = np.array(y_10_test_10)
   X_10_test_15 = np.array(X_10_test_15)
   y_10_test_15 = np.array(y_10_test_15)
   X_10_test_20 = np.array(X_10_test_20)
   y_10_test_20 = np.array(y_10_test_20)

   X_10_train = torch.tensor(X_10_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_10_train = torch.tensor(y_10_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   X_10_test_5 = torch.tensor(X_10_test_5, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_10_test_5 = torch.tensor(y_10_test_5, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_10_test_10 = torch.tensor(X_10_test_10, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_10_test_10 = torch.tensor(y_10_test_10, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_10_test_15 = torch.tensor(X_10_test_15, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_10_test_15 = torch.tensor(y_10_test_15, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_10_test_20 = torch.tensor(X_10_test_20, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_10_test_20 = torch.tensor(y_10_test_20, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_10_train = torch.nan_to_num(X_10_train, nan=0.0)
   X_10_test_5 = torch.nan_to_num(X_10_test_5, nan=0.0)
   X_10_test_10 = torch.nan_to_num(X_10_test_10, nan=0.0)
   X_10_test_15 = torch.nan_to_num(X_10_test_15, nan=0.0)
   X_10_test_20 = torch.nan_to_num(X_10_test_20, nan=0.0)


   # tensorizing the data for 15
   X_15_train = np.array(X_15_train)
   y_15_train = np.array(y_15_train)
   X_15_test_5 = np.array(X_15_test_5)
   y_15_test_5 = np.array(y_15_test_5)
   X_15_test_10 = np.array(X_15_test_10)
   y_15_test_10 = np.array(y_15_test_10)
   X_15_test_15 = np.array(X_15_test_15)
   y_15_test_15 = np.array(y_15_test_15)
   X_15_test_20 = np.array(X_15_test_20)
   y_15_test_20 = np.array(y_15_test_20)

   X_15_train = torch.tensor(X_15_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_15_train = torch.tensor(y_15_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   X_15_test_5 = torch.tensor(X_15_test_5, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_15_test_5 = torch.tensor(y_15_test_5, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_15_test_10 = torch.tensor(X_15_test_10, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_15_test_10 = torch.tensor(y_15_test_10, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_15_test_15 = torch.tensor(X_15_test_15, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_15_test_15 = torch.tensor(y_15_test_15, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_15_test_20 = torch.tensor(X_15_test_20, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_15_test_20 = torch.tensor(y_15_test_20, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_15_train = torch.nan_to_num(X_15_train, nan=0.0)
   X_15_test_5 = torch.nan_to_num(X_15_test_5, nan=0.0)
   X_15_test_10 = torch.nan_to_num(X_15_test_10, nan=0.0)
   X_15_test_15 = torch.nan_to_num(X_15_test_15, nan=0.0)
   X_15_test_20 = torch.nan_to_num(X_15_test_20, nan=0.0)



   # tensorizing the data for 20
   X_20_train = np.array(X_20_train)
   y_20_train = np.array(y_20_train)
   X_20_test_5 = np.array(X_20_test_5)
   y_20_test_5 = np.array(y_20_test_5)
   X_20_test_10 = np.array(X_20_test_10)
   y_20_test_10 = np.array(y_20_test_10)
   X_20_test_15 = np.array(X_20_test_15)
   y_20_test_15 = np.array(y_20_test_15)
   X_20_test_20 = np.array(X_20_test_20)
   y_20_test_20 = np.array(y_20_test_20)

   X_20_train = torch.tensor(X_20_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_20_train = torch.tensor(y_20_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   X_20_test_5 = torch.tensor(X_20_test_5, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_20_test_5 = torch.tensor(y_20_test_5, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_20_test_10 = torch.tensor(X_20_test_10, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_20_test_10 = torch.tensor(y_20_test_10, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_20_test_15 = torch.tensor(X_20_test_15, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_20_test_15 = torch.tensor(y_20_test_15, dtype = torch.long)     # Shape will be (num_samples, 1)
   X_20_test_20 = torch.tensor(X_20_test_20, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_20_test_20 = torch.tensor(y_20_test_20, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_20_train = torch.nan_to_num(X_20_train, nan=0.0)
   X_20_test_5 = torch.nan_to_num(X_20_test_5, nan=0.0)
   X_20_test_10 = torch.nan_to_num(X_20_test_10, nan=0.0)
   X_20_test_15 = torch.nan_to_num(X_20_test_15, nan=0.0)
   X_20_test_20 = torch.nan_to_num(X_20_test_20, nan=0.0)



   # ----------------------------------------
   # ----------------------------------------
   # ----------------------------------------

   
   # Reshape your input to add sequence length dimension for 5
   X_5_train = X_5_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_5_test_5 = X_5_test_5.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_5_test_10 = X_5_test_10.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_5_test_15 = X_5_test_15.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_5_test_20 = X_5_test_20.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]

   train_dataset_5 = TensorDataset(X_5_train, y_5_train)
   test_dataset_5_5 = TensorDataset(X_5_test_5, y_5_test_5)
   test_dataset_5_10 = TensorDataset(X_5_test_10, y_5_test_10)
   test_dataset_5_15 = TensorDataset(X_5_test_15, y_5_test_15)
   test_dataset_5_20 = TensorDataset(X_5_test_20, y_5_test_20)


   # Reshape your input to add sequence length dimension for 10
   X_10_train = X_10_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_10_test_5 = X_10_test_5.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_10_test_10 = X_10_test_10.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_10_test_15 = X_10_test_15.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_10_test_20 = X_10_test_20.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]

   train_dataset_10 = TensorDataset(X_10_train, y_10_train)
   test_dataset_10_5 = TensorDataset(X_10_test_5, y_10_test_5)
   test_dataset_10_10 = TensorDataset(X_10_test_10, y_10_test_10)
   test_dataset_10_15 = TensorDataset(X_10_test_15, y_10_test_15)
   test_dataset_10_20 = TensorDataset(X_10_test_20, y_10_test_20)


   # Reshape your input to add sequence length dimension for 15
   X_15_train = X_15_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_15_test_5 = X_15_test_5.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_15_test_10 = X_15_test_10.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_15_test_15 = X_15_test_15.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_15_test_20 = X_15_test_20.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]

   train_dataset_15 = TensorDataset(X_15_train, y_15_train)
   test_dataset_15_5 = TensorDataset(X_15_test_5, y_15_test_5)
   test_dataset_15_10 = TensorDataset(X_15_test_10, y_15_test_10)
   test_dataset_15_15 = TensorDataset(X_15_test_15, y_15_test_15)
   test_dataset_15_20 = TensorDataset(X_15_test_20, y_15_test_20)


   # Reshape your input to add sequence length dimension for 20
   X_20_train = X_20_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_20_test_5 = X_20_test_5.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_20_test_10 = X_20_test_10.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_20_test_15 = X_20_test_15.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   X_20_test_20 = X_20_test_20.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]

   train_dataset_20 = TensorDataset(X_20_train, y_20_train)
   test_dataset_20_5 = TensorDataset(X_20_test_5, y_20_test_5)
   test_dataset_20_10 = TensorDataset(X_20_test_10, y_20_test_10)
   test_dataset_20_15 = TensorDataset(X_20_test_15, y_20_test_15)
   test_dataset_20_20 = TensorDataset(X_20_test_20, y_20_test_20)



   # ----------------------------------------
   # ----------------------------------------
   # ----------------------------------------

   # Create DataLoaders for train and test sets 5
   train_loader_5 = DataLoader(train_dataset_5, batch_size=batch_size, shuffle=True)
   test_loader_5_5 = DataLoader(test_dataset_5_5, batch_size=batch_size, shuffle=False)
   test_loader_5_10 = DataLoader(test_dataset_5_10, batch_size=batch_size, shuffle=False)
   test_loader_5_15 = DataLoader(test_dataset_5_15, batch_size=batch_size, shuffle=False)
   test_loader_5_20 = DataLoader(test_dataset_5_20, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 10
   train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True)
   test_loader_10_5 = DataLoader(test_dataset_10_5, batch_size=batch_size, shuffle=False)
   test_loader_10_10 = DataLoader(test_dataset_10_10, batch_size=batch_size, shuffle=False)
   test_loader_10_15 = DataLoader(test_dataset_10_15, batch_size=batch_size, shuffle=False)
   test_loader_10_20 = DataLoader(test_dataset_10_20, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 15
   train_loader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True)
   test_loader_15_5 = DataLoader(test_dataset_15_5, batch_size=batch_size, shuffle=False)
   test_loader_15_10 = DataLoader(test_dataset_15_10, batch_size=batch_size, shuffle=False)
   test_loader_15_15 = DataLoader(test_dataset_15_15, batch_size=batch_size, shuffle=False)
   test_loader_15_20 = DataLoader(test_dataset_15_20, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 20
   train_loader_20 = DataLoader(train_dataset_20, batch_size=batch_size, shuffle=True)
   test_loader_20_5 = DataLoader(test_dataset_20_5, batch_size=batch_size, shuffle=False)
   test_loader_20_10 = DataLoader(test_dataset_20_10, batch_size=batch_size, shuffle=False)
   test_loader_20_15 = DataLoader(test_dataset_20_15, batch_size=batch_size, shuffle=False)
   test_loader_20_20 = DataLoader(test_dataset_20_20, batch_size=batch_size, shuffle=False)


   # ----------------------------------------
   # ----------------------------------------
   # ----------------------------------------

   # define 4 IDS models
   model_5 = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_10 = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_15 = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_20 = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

   # Train and Test 5
   pr.prGreen("Start Training!")
   model_5.model_train(epochs = epochs,train_loader = train_loader_5)
   pr.prGreen("__ 5 Trained!")
   model_5.check_model_nans(test_loader_5_5)
   model_5.check_model_nans(test_loader_5_10)
   model_5.check_model_nans(test_loader_5_15)
   model_5.check_model_nans(test_loader_5_20)
   roc_auc_5_5.append(round(model_5.evaluate_model_ROCAUC(test_loader_5_5),3))
   roc_auc_5_10.append(round(model_5.evaluate_model_ROCAUC(test_loader_5_10),3))
   roc_auc_5_15.append(round(model_5.evaluate_model_ROCAUC(test_loader_5_15),3))
   roc_auc_5_20.append(round(model_5.evaluate_model_ROCAUC(test_loader_5_20),3))
   

   
   # Train and Test 10
   pr.prGreen("Start Training!")
   model_10.model_train(epochs = epochs,train_loader = train_loader_10)
   pr.prGreen("__ 10 Trained!")
   model_10.check_model_nans(test_loader_10_5)
   model_10.check_model_nans(test_loader_10_10)
   model_10.check_model_nans(test_loader_10_15)
   model_10.check_model_nans(test_loader_10_20)
   roc_auc_10_5.append(round(model_10.evaluate_model_ROCAUC(test_loader_10_5),3))
   roc_auc_10_10.append(round(model_10.evaluate_model_ROCAUC(test_loader_10_10),3))
   roc_auc_10_15.append(round(model_10.evaluate_model_ROCAUC(test_loader_10_15),3))
   roc_auc_10_20.append(round(model_10.evaluate_model_ROCAUC(test_loader_10_20),3))
   


   # Train and Test 15
   pr.prGreen("Start Training!")
   model_15.model_train(epochs = epochs,train_loader = train_loader_15)
   pr.prGreen("__ 15 Trained!")
   model_15.check_model_nans(test_loader_15_5)
   model_15.check_model_nans(test_loader_15_10)
   model_15.check_model_nans(test_loader_15_15)
   model_15.check_model_nans(test_loader_15_20)
   roc_auc_15_5.append(round(model_15.evaluate_model_ROCAUC(test_loader_15_5),3))
   roc_auc_15_10.append(round(model_15.evaluate_model_ROCAUC(test_loader_15_10),3))
   roc_auc_15_15.append(round(model_15.evaluate_model_ROCAUC(test_loader_15_15),3))
   roc_auc_15_20.append(round(model_15.evaluate_model_ROCAUC(test_loader_15_20),3))


   # Train and Test 20
   pr.prGreen("Start Training!")
   model_20.model_train(epochs = epochs,train_loader = train_loader_20)
   pr.prGreen("__ 20 Trained!")
   model_20.check_model_nans(test_loader_20_5)
   model_20.check_model_nans(test_loader_20_10)
   model_20.check_model_nans(test_loader_20_15)
   model_20.check_model_nans(test_loader_20_20)
   roc_auc_20_5.append(round(model_20.evaluate_model_ROCAUC(test_loader_20_5),3))
   roc_auc_20_10.append(round(model_20.evaluate_model_ROCAUC(test_loader_20_10),3))
   roc_auc_20_15.append(round(model_20.evaluate_model_ROCAUC(test_loader_20_15),3))
   roc_auc_20_20.append(round(model_20.evaluate_model_ROCAUC(test_loader_20_20),3))

   

   
# save data
with open(add + '/roc_auc_5_5.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_5_5)

with open(add + '/roc_auc_5_10.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_5_10)

with open(add + '/roc_auc_5_15.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_5_15)

with open(add + '/roc_auc_5_20.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_5_20)

with open(add + '/roc_auc_10_5.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_10_5)

with open(add + '/roc_auc_10_10.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_10_10)

with open(add + '/roc_auc_10_15.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_10_15)

with open(add + '/roc_auc_10_20.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_10_20)

with open(add + '/roc_auc_15_5.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_15_5)

with open(add + '/roc_auc_15_10.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_15_10)

with open(add + '/roc_auc_15_15.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_15_15)

with open(add + '/roc_auc_15_20.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_15_20)

with open(add + '/roc_auc_20_5.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_20_5)

with open(add + '/roc_auc_20_10.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_20_10)

with open(add + '/roc_auc_20_15.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_20_15)

with open(add + '/roc_auc_20_20.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_20_20)

