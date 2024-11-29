#!/usr/bin/env python3

import myLSTM as mine
import printing as pr
import seqMaker
import observableToSink as obs
import pandas as pd
import numpy as np
import os
import time
import re
import aggregate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import depth
import argparse
import csv
import LSTM_FED



"""
In this scenario we train a global IDS model using the horizontal Federated Learning method.
We test attack detection accuracy over time. The evaluation is over data from networks with size 5, 10, 15 and 20.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime (FL): network size (5, 10, 15, 20)")

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




for run in range(runs):
   pr.prGreen("run " + str(run))
   print(".................................")
   _5_Train, _5_Test = aggregate.aggregate_list_5()
   _10_Train, _10_Test = aggregate.aggregate_list_10()
   _15_Train, _15_Test = aggregate.aggregate_list_15()
   _20_Train, _20_Test = aggregate.aggregate_list_20()



   ##   5
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
   #__5_normalized_dfs_test = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _5_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   ##   10
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
   #__10_normalized_dfs_test = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _10_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   ##   15
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
   #__15_normalized_dfs_test = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _15_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   ##   20
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
   #__20_normalized_dfs_test = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _20_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   __5_normalized_dfs_test_5 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _5_Test]
   transition_indices_5_5 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __5_normalized_dfs_test_5]
   len_5_5 = [df.shape[0] for df in __5_normalized_dfs_test_5]

   __5_normalized_dfs_test_10 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _10_Test]
   transition_indices_5_10 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __5_normalized_dfs_test_10]
   len_5_10 = [df.shape[0] for df in __5_normalized_dfs_test_10]

   __5_normalized_dfs_test_15 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _15_Test]
   transition_indices_5_15 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __5_normalized_dfs_test_15]
   len_5_15 = [df.shape[0] for df in __5_normalized_dfs_test_15]

   __5_normalized_dfs_test_20 = [df.apply(lambda x: (x - __5_min[x.name]) / (__5_max[x.name] - __5_min[x.name])) for df in _20_Test]
   transition_indices_5_20 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __5_normalized_dfs_test_20]
   len_5_20 = [df.shape[0] for df in __5_normalized_dfs_test_20]

   #

   __10_normalized_dfs_test_5 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _5_Test]    
   transition_indices_10_5 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __10_normalized_dfs_test_5]
   len_10_5 = [df.shape[0] for df in __10_normalized_dfs_test_5]

   __10_normalized_dfs_test_10 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _10_Test]
   transition_indices_10_10 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __10_normalized_dfs_test_10]
   len_10_10 = [df.shape[0] for df in __10_normalized_dfs_test_10]

   __10_normalized_dfs_test_15 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _15_Test]
   transition_indices_10_15 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __10_normalized_dfs_test_15]
   len_10_15 = [df.shape[0] for df in __10_normalized_dfs_test_15]

   __10_normalized_dfs_test_20 = [df.apply(lambda x: (x - __10_min[x.name]) / (__10_max[x.name] - __10_min[x.name])) for df in _20_Test]
   transition_indices_10_20 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __10_normalized_dfs_test_20]
   len_10_20 = [df.shape[0] for df in __10_normalized_dfs_test_20]

   #

   __15_normalized_dfs_test_5 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _5_Test]
   transition_indices_15_5 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __15_normalized_dfs_test_5]
   len_15_5 = [df.shape[0] for df in __15_normalized_dfs_test_5]

   __15_normalized_dfs_test_10 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _10_Test]
   transition_indices_15_10 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __15_normalized_dfs_test_10]
   len_15_10 = [df.shape[0] for df in __15_normalized_dfs_test_10]

   __15_normalized_dfs_test_15 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _15_Test]
   transition_indices_15_15 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __15_normalized_dfs_test_15]
   len_15_15 = [df.shape[0] for df in __15_normalized_dfs_test_15]

   __15_normalized_dfs_test_20 = [df.apply(lambda x: (x - __15_min[x.name]) / (__15_max[x.name] - __15_min[x.name])) for df in _20_Test]
   transition_indices_15_20 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __15_normalized_dfs_test_20]
   len_15_20 = [df.shape[0] for df in __15_normalized_dfs_test_20]

   #

   __20_normalized_dfs_test_5 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _5_Test]    
   transition_indices_20_5 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __20_normalized_dfs_test_5]
   len_20_5 = [df.shape[0] for df in __20_normalized_dfs_test_5]

   __20_normalized_dfs_test_10 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _10_Test]
   transition_indices_20_10 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __20_normalized_dfs_test_10]
   len_20_10 = [df.shape[0] for df in __20_normalized_dfs_test_10]

   __20_normalized_dfs_test_15 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _15_Test]
   transition_indices_20_15 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __20_normalized_dfs_test_15]
   len_20_15 = [df.shape[0] for df in __20_normalized_dfs_test_15]

   __20_normalized_dfs_test_20 = [df.apply(lambda x: (x - __20_min[x.name]) / (__20_max[x.name] - __20_min[x.name])) for df in _20_Test]
   transition_indices_20_20 = [df['label'].apply(lambda x: x == 1).idxmax() for df in __20_normalized_dfs_test_20]
   len_20_20 = [df.shape[0] for df in __20_normalized_dfs_test_20]



   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   
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
   sequenced_10_Train = pd.concat(sequenced_10_Train, ignore_index=True)
   sequenced_15_Train = pd.concat(sequenced_15_Train, ignore_index=True)
   sequenced_20_Train = pd.concat(sequenced_20_Train, ignore_index=True)
   
   
   # ------------------------------------------------------------------------------------------------------------ 

   # extract X and y 
   X_5_train = sequenced_5_Train.iloc[:, :-1].values  # All columns except the last one
   y_5_train = sequenced_5_Train.iloc[:, -1].values  # The last column
   X_10_train = sequenced_10_Train.iloc[:, :-1].values  # All columns except the last one
   y_10_train = sequenced_10_Train.iloc[:, -1].values  # The last column
   X_15_train = sequenced_15_Train.iloc[:, :-1].values  # All columns except the last one
   y_15_train = sequenced_15_Train.iloc[:, -1].values  # The last column
   X_20_train = sequenced_20_Train.iloc[:, :-1].values  # All columns except the last one
   y_20_train = sequenced_20_Train.iloc[:, -1].values  # The last column


   
   

   # ----------------------------------------
   # tensorizing the data for 5
   X_5_train = np.array(X_5_train)
   y_5_train = np.array(y_5_train)

   X_5_train = torch.tensor(X_5_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_5_train = torch.tensor(y_5_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_5_train = torch.nan_to_num(X_5_train, nan=0.0)
   #y_5_train = torch.nan_to_num(y_5_train, nan=0.0)
   


   # tensorizing the data for 10
   X_10_train = np.array(X_10_train)
   y_10_train = np.array(y_10_train)
   
   X_10_train = torch.tensor(X_10_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_10_train = torch.tensor(y_10_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_10_train = torch.nan_to_num(X_10_train, nan=0.0)
   #y_5_train = torch.nan_to_num(y_5_train, nan=0.0)
   

   # tensorizing the data for 15
   X_15_train = np.array(X_15_train)
   y_15_train = np.array(y_15_train)

   X_15_train = torch.tensor(X_15_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_15_train = torch.tensor(y_15_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_15_train = torch.nan_to_num(X_15_train, nan=0.0)
   #y_5_train = torch.nan_to_num(y_5_train, nan=0.0)
   


   # tensorizing the data for 20
   X_20_train = np.array(X_20_train)
   y_20_train = np.array(y_20_train)
   
   X_20_train = torch.tensor(X_20_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_20_train = torch.tensor(y_20_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_20_train = torch.nan_to_num(X_20_train, nan=0.0)
   #y_5_train = torch.nan_to_num(y_5_train, nan=0.0)
   
   # ------------------------------------------------
   # Reshape your input to add sequence length dimension for 5
   X_5_train = X_5_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_5 = TensorDataset(X_5_train, y_5_train)

   # Reshape your input to add sequence length dimension for 10
   X_10_train = X_10_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_10 = TensorDataset(X_10_train, y_10_train)

   # Reshape your input to add sequence length dimension for 15
   X_15_train = X_15_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_15 = TensorDataset(X_15_train, y_15_train)


   # Reshape your input to add sequence length dimension for 20
   X_20_train = X_20_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_20 = TensorDataset(X_20_train, y_20_train)
   # ------------------------------------------
   # Create DataLoaders for train and test sets 5
   train_loader_5 = DataLoader(train_dataset_5, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 10
   train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 15
   train_loader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 20
   train_loader_20 = DataLoader(train_dataset_20, batch_size=batch_size, shuffle=True)

   # ------------------------------------------

   # Initialize client A and client B's models
   client_5 = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_10 = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_15 = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_20 = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
                      
   clients = [client_5, client_10, client_15, client_20]
   training_loaders = [train_loader_5, train_loader_10, train_loader_15, train_loader_20]


   server_model = LSTM_FED.federated_training(clients[0], clients, global_epochs = rnds, client_epochs = epochs_per_client, training_loaders = training_loaders)

   # ------------------------------------------------------------------------------------------------------------
   # ------------------------------------------------------------------------------------------------------------
   def dataPreparation(df):
      # extract X, y
      X = df.iloc[:, :-1].values  # All columns except the last one
      y = df.iloc[:, -1].values  # The last column

      # tensorizing the data 
      X = np.array(X)
      y = np.array(y)
      X = torch.tensor(X, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
      y = torch.tensor(y, dtype = torch.long)     # Shape will be (num_samples, 1)

      # Check for NaNs due to min-max normalization
      X = torch.nan_to_num(X, nan=0.0)

      # Reshape your input to add sequence length dimension
      X = X.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
      myDataset = TensorDataset(X, y)

      # Create DataLoaders for test sets bh
      myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=False)

      return myDataLoader


   def timeFinder(dl, trans):
      #print(".....................................................")
      for inputs, label in dl:
         detection = torch.argmax(server_model(inputs), dim=1)
         sliced_detection = detection[trans:]

         # Find the index of the first occurrence of 1 in the slice
         first_one_relative = torch.nonzero(sliced_detection == 1, as_tuple=True)[0]
      
         if len(first_one_relative) > 0:
            detectionTime = first_one_relative[0].item()  # Adjust for the original index
            
         else:
            detectionTime = -1

         return detectionTime

   # ------------------------------------------------------------------------------------------------------------
   # ------------------------------------------------------------------------------------------------------------

   # Test over each seq:

   # test 5 on the global server after normalized on 5/10/15/20
   _5_5CoojaRuns = [dataPreparation(df) for df in sequenced_5_Test_5]
   _10_5CoojaRuns = [dataPreparation(df) for df in sequenced_10_Test_5]
   _15_5CoojaRuns = [dataPreparation(df) for df in sequenced_15_Test_5]
   _20_5CoojaRuns = [dataPreparation(df) for df in sequenced_20_Test_5]

   # test 10 on the global server after normalized on 5/10/15/20
   _5_10CoojaRuns = [dataPreparation(df) for df in sequenced_5_Test_10]
   _10_10CoojaRuns = [dataPreparation(df) for df in sequenced_10_Test_10]
   _15_10CoojaRuns = [dataPreparation(df) for df in sequenced_15_Test_10]
   _20_10CoojaRuns = [dataPreparation(df) for df in sequenced_20_Test_10]

   # test 15 on the global server after normalized on 5/10/15/20
   _5_15CoojaRuns = [dataPreparation(df) for df in sequenced_5_Test_15]
   _10_15CoojaRuns = [dataPreparation(df) for df in sequenced_10_Test_15]
   _15_15CoojaRuns = [dataPreparation(df) for df in sequenced_15_Test_15]
   _20_15CoojaRuns = [dataPreparation(df) for df in sequenced_20_Test_15]

   # test 20 on the global server after normalized on 5/10/15/20
   _5_20CoojaRuns = [dataPreparation(df) for df in sequenced_5_Test_20]
   _10_20CoojaRuns = [dataPreparation(df) for df in sequenced_10_Test_20]
   _15_20CoojaRuns = [dataPreparation(df) for df in sequenced_15_Test_20]
   _20_20CoojaRuns = [dataPreparation(df) for df in sequenced_20_Test_20]

   
   

   # 
   len5_5CoojaRuns = [len(dl.dataset) for dl in _5_5CoojaRuns]
   len10_5CoojaRuns = [len(dl.dataset) for dl in _10_5CoojaRuns]
   len15_5CoojaRuns = [len(dl.dataset) for dl in _15_5CoojaRuns]
   len120_5CoojaRuns = [len(dl.dataset) for dl in _20_5CoojaRuns]

   # 
   len5_10CoojaRuns = [len(dl.dataset) for dl in _5_10CoojaRuns]
   len10_10CoojaRuns = [len(dl.dataset) for dl in _10_10CoojaRuns]
   len15_10CoojaRuns = [len(dl.dataset) for dl in _15_10CoojaRuns]
   len120_10CoojaRuns = [len(dl.dataset) for dl in _20_10CoojaRuns]

   # 
   len5_15CoojaRuns = [len(dl.dataset) for dl in _5_15CoojaRuns]
   len10_15CoojaRuns = [len(dl.dataset) for dl in _10_15CoojaRuns]
   len15_15CoojaRuns = [len(dl.dataset) for dl in _15_15CoojaRuns]
   len120_15CoojaRuns = [len(dl.dataset) for dl in _20_15CoojaRuns]

   # 
   len5_20CoojaRuns = [len(dl.dataset) for dl in _5_20CoojaRuns]
   len10_20CoojaRuns = [len(dl.dataset) for dl in _10_20CoojaRuns]
   len15_20CoojaRuns = [len(dl.dataset) for dl in _15_20CoojaRuns]
   len120_20CoojaRuns = [len(dl.dataset) for dl in _20_20CoojaRuns]


   #
   _5_5_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_5_5CoojaRuns, transition_indices_5_5)]
   _10_5_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_10_5CoojaRuns, transition_indices_10_5)]
   _15_5_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_15_5CoojaRuns, transition_indices_15_5)]
   _20_5_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_20_5CoojaRuns, transition_indices_20_5)]

   _5_10_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_5_10CoojaRuns, transition_indices_5_10)]
   _10_10_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_10_10CoojaRuns, transition_indices_10_10)]
   _15_10_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_15_10CoojaRuns, transition_indices_15_10)]
   _20_10_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_20_10CoojaRuns, transition_indices_20_10)]

   _5_15_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_5_15CoojaRuns, transition_indices_5_15)]
   _10_15_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_10_15CoojaRuns, transition_indices_10_15)]
   _15_15_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_15_15CoojaRuns, transition_indices_15_15)]
   _20_15_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_20_15CoojaRuns, transition_indices_20_15)]

   _5_20_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_5_20CoojaRuns, transition_indices_5_20)]
   _10_20_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_10_20CoojaRuns, transition_indices_10_20)]
   _15_20_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_15_20CoojaRuns, transition_indices_15_20)]
   _20_20_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_20_20CoojaRuns, transition_indices_20_20)]


   # bh_dis  means dis test that ius normalized by bh data             
   with open(add + '/detect_time_fed_5_5.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_5_5_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_10_5.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_10_5_CoojaRunPrediction)

   with open(add + '/detect_time_fed_15_5.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_15_5_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_20_5.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_20_5_CoojaRunPrediction)


   with open(add + '/detect_time_fed_5_10.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_5_10_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_10_10.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_10_10_CoojaRunPrediction)

   with open(add + '/detect_time_fed_15_10.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_15_10_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_20_10.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_20_10_CoojaRunPrediction)


   with open(add + '/detect_time_fed_5_15.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_5_15_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_10_15.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_10_15_CoojaRunPrediction)

   with open(add + '/detect_time_fed_15_15.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_15_15_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_20_15.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_20_15_CoojaRunPrediction)


   with open(add + '/detect_time_fed_5_20.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_5_20_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_10_20.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_10_20_CoojaRunPrediction)

   with open(add + '/detect_time_fed_15_20.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_15_20_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_20_20.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_20_20_CoojaRunPrediction)

   

   