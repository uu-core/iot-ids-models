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
We test attack detection accuracy over time. The data that are used for testing are from the shallow, medium and deep topologies.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime (FL): network topologies (shallow, medium, deep)")

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
   _shallow_Train, _shallow_Test = aggregate.aggregate_list_shallow()
   _medium_Train, _medium_Test = aggregate.aggregate_list_medium()
   _deep_Train, _deep_Test = aggregate.aggregate_list_deep()



   ##   5
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _shallow_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __shallow_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __shallow_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __shallow_max = __shallow_all_maxs_df.max(axis = 0)
   __shallow_min = __shallow_all_mins_df.min(axis = 0)
   __shallow_normalized_dfs_train = [df.apply(lambda x: (x - __shallow_min[x.name]) / (__shallow_max[x.name] - __shallow_min[x.name])) for df in _shallow_Train]
   #__shallow_normalized_dfs_test = [df.apply(lambda x: (x - __shallow_min[x.name]) / (__shallow_max[x.name] - __shallow_min[x.name])) for df in _shallow_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   ##   10
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _medium_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __medium_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __medium_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __medium_max = __medium_all_maxs_df.max(axis = 0)
   __medium_min = __medium_all_mins_df.min(axis = 0)
   __medium_normalized_dfs_train = [df.apply(lambda x: (x - __medium_min[x.name]) / (__medium_max[x.name] - __medium_min[x.name])) for df in _medium_Train]
   #__medium_normalized_dfs_test = [df.apply(lambda x: (x - __medium_min[x.name]) / (__medium_max[x.name] - __medium_min[x.name])) for df in _medium_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   ##   15
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _deep_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __deep_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __deep_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __deep_max = __deep_all_maxs_df.max(axis = 0)
   __deep_min = __deep_all_mins_df.min(axis = 0)
   __deep_normalized_dfs_train = [df.apply(lambda x: (x - __deep_min[x.name]) / (__deep_max[x.name] - __deep_min[x.name])) for df in _deep_Train]
   #__deep_normalized_dfs_test = [df.apply(lambda x: (x - __deep_min[x.name]) / (__deep_max[x.name] - __deep_min[x.name])) for df in _deep_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   __shallow_normalized_dfs_test_shallow = [df.apply(lambda x: (x - __shallow_min[x.name]) / (__shallow_max[x.name] - __shallow_min[x.name])) for df in _shallow_Test]
   transition_indices_shallow_shallow = [df['label'].apply(lambda x: x == 1).idxmax() for df in __shallow_normalized_dfs_test_shallow]
   len_shallow_shallow = [df.shape[0] for df in __shallow_normalized_dfs_test_shallow]

   __shallow_normalized_dfs_test_medium = [df.apply(lambda x: (x - __shallow_min[x.name]) / (__shallow_max[x.name] - __shallow_min[x.name])) for df in _medium_Test]
   transition_indices_shallow_medium = [df['label'].apply(lambda x: x == 1).idxmax() for df in __shallow_normalized_dfs_test_medium]
   len_shallow_medium = [df.shape[0] for df in __shallow_normalized_dfs_test_medium]

   __shallow_normalized_dfs_test_deep = [df.apply(lambda x: (x - __shallow_min[x.name]) / (__shallow_max[x.name] - __shallow_min[x.name])) for df in _deep_Test]
   transition_indices_shallow_deep = [df['label'].apply(lambda x: x == 1).idxmax() for df in __shallow_normalized_dfs_test_deep]
   len_shallow_deep = [df.shape[0] for df in __shallow_normalized_dfs_test_deep]

   #

   __medium_normalized_dfs_test_shallow = [df.apply(lambda x: (x - __medium_min[x.name]) / (__medium_max[x.name] - __medium_min[x.name])) for df in _shallow_Test]    
   transition_indices_medium_shallow = [df['label'].apply(lambda x: x == 1).idxmax() for df in __medium_normalized_dfs_test_shallow]
   len_medium_shallow = [df.shape[0] for df in __medium_normalized_dfs_test_shallow]

   __medium_normalized_dfs_test_medium = [df.apply(lambda x: (x - __medium_min[x.name]) / (__medium_max[x.name] - __medium_min[x.name])) for df in _medium_Test]
   transition_indices_medium_medium = [df['label'].apply(lambda x: x == 1).idxmax() for df in __medium_normalized_dfs_test_medium]
   len_medium_medium = [df.shape[0] for df in __medium_normalized_dfs_test_medium]

   __medium_normalized_dfs_test_deep = [df.apply(lambda x: (x - __medium_min[x.name]) / (__medium_max[x.name] - __medium_min[x.name])) for df in _deep_Test]
   transition_indices_medium_deep = [df['label'].apply(lambda x: x == 1).idxmax() for df in __medium_normalized_dfs_test_deep]
   len_medium_deep = [df.shape[0] for df in __medium_normalized_dfs_test_deep]

   #

   __deep_normalized_dfs_test_shallow = [df.apply(lambda x: (x - __deep_min[x.name]) / (__deep_max[x.name] - __deep_min[x.name])) for df in _shallow_Test]
   transition_indices_deep_shallow = [df['label'].apply(lambda x: x == 1).idxmax() for df in __deep_normalized_dfs_test_shallow]
   len_deep_shallow = [df.shape[0] for df in __deep_normalized_dfs_test_shallow]

   __deep_normalized_dfs_test_medium = [df.apply(lambda x: (x - __deep_min[x.name]) / (__deep_max[x.name] - __deep_min[x.name])) for df in _medium_Test]
   transition_indices_deep_medium = [df['label'].apply(lambda x: x == 1).idxmax() for df in __deep_normalized_dfs_test_medium]
   len_deep_medium = [df.shape[0] for df in __deep_normalized_dfs_test_medium]

   __deep_normalized_dfs_test_deep = [df.apply(lambda x: (x - __deep_min[x.name]) / (__deep_max[x.name] - __deep_min[x.name])) for df in _deep_Test]
   transition_indices_deep_deep = [df['label'].apply(lambda x: x == 1).idxmax() for df in __deep_normalized_dfs_test_deep]
   len_deep_deep = [df.shape[0] for df in __deep_normalized_dfs_test_deep]





   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   
   sequenced_shallow_Train = [seqMaker.seq_maker(df,10) for df in __shallow_normalized_dfs_train]
   sequenced_shallow_Test_shallow = [seqMaker.seq_maker(df,10) for df in __shallow_normalized_dfs_test_shallow]
   sequenced_shallow_Test_medium = [seqMaker.seq_maker(df,10) for df in __shallow_normalized_dfs_test_medium]
   sequenced_shallow_Test_deep = [seqMaker.seq_maker(df,10) for df in __shallow_normalized_dfs_test_deep]

   sequenced_medium_Train = [seqMaker.seq_maker(df,10) for df in __medium_normalized_dfs_train]
   sequenced_medium_Test_shallow = [seqMaker.seq_maker(df,10) for df in __medium_normalized_dfs_test_shallow]
   sequenced_medium_Test_medium = [seqMaker.seq_maker(df,10) for df in __medium_normalized_dfs_test_medium]
   sequenced_medium_Test_deep = [seqMaker.seq_maker(df,10) for df in __medium_normalized_dfs_test_deep]

   sequenced_deep_Train = [seqMaker.seq_maker(df,10) for df in __deep_normalized_dfs_train]
   sequenced_deep_Test_shallow = [seqMaker.seq_maker(df,10) for df in __deep_normalized_dfs_test_shallow]
   sequenced_deep_Test_medium = [seqMaker.seq_maker(df,10) for df in __deep_normalized_dfs_test_medium]
   sequenced_deep_Test_deep = [seqMaker.seq_maker(df,10) for df in __deep_normalized_dfs_test_deep]


   # 
   sequenced_shallow_Train = pd.concat(sequenced_shallow_Train, ignore_index=True)
   sequenced_medium_Train = pd.concat(sequenced_medium_Train, ignore_index=True)
   sequenced_deep_Train = pd.concat(sequenced_deep_Train, ignore_index=True)
   
   
   # ------------------------------------------------------------------------------------------------------------ 

   # extract X and y 
   X_shallow_train = sequenced_shallow_Train.iloc[:, :-1].values  # All columns except the last one
   y_shallow_train = sequenced_shallow_Train.iloc[:, -1].values  # The last column
   X_medium_train = sequenced_medium_Train.iloc[:, :-1].values  # All columns except the last one
   y_medium_train = sequenced_medium_Train.iloc[:, -1].values  # The last column
   X_deep_train = sequenced_deep_Train.iloc[:, :-1].values  # All columns except the last one
   y_deep_train = sequenced_deep_Train.iloc[:, -1].values  # The last column
   

   # ----------------------------------------
   # tensorizing the data for 5
   X_shallow_train = np.array(X_shallow_train)
   y_shallow_train = np.array(y_shallow_train)

   X_shallow_train = torch.tensor(X_shallow_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_shallow_train = torch.tensor(y_shallow_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_shallow_train = torch.nan_to_num(X_shallow_train, nan=0.0)
   #y_shallow_train = torch.nan_to_num(y_shallow_train, nan=0.0)
   


   # tensorizing the data for 10
   X_medium_train = np.array(X_medium_train)
   y_medium_train = np.array(y_medium_train)
   
   X_medium_train = torch.tensor(X_medium_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_medium_train = torch.tensor(y_medium_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_medium_train = torch.nan_to_num(X_medium_train, nan=0.0)
   #y_shallow_train = torch.nan_to_num(y_shallow_train, nan=0.0)
   

   # tensorizing the data for 15
   X_deep_train = np.array(X_deep_train)
   y_deep_train = np.array(y_deep_train)

   X_deep_train = torch.tensor(X_deep_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_deep_train = torch.tensor(y_deep_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_deep_train = torch.nan_to_num(X_deep_train, nan=0.0)
   #y_shallow_train = torch.nan_to_num(y_shallow_train, nan=0.0)

   
   # ------------------------------------------------
   # Reshape your input to add sequence length dimension for 5
   X_shallow_train = X_shallow_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_shallow = TensorDataset(X_shallow_train, y_shallow_train)

   # Reshape your input to add sequence length dimension for 10
   X_medium_train = X_medium_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_medium = TensorDataset(X_medium_train, y_medium_train)

   # Reshape your input to add sequence length dimension for 15
   X_deep_train = X_deep_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_deep = TensorDataset(X_deep_train, y_deep_train)

   # ------------------------------------------
   # Create DataLoaders for train and test sets 5
   train_loader_shallow = DataLoader(train_dataset_shallow, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 10
   train_loader_medium = DataLoader(train_dataset_medium, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 15
   train_loader_deep = DataLoader(train_dataset_deep, batch_size=batch_size, shuffle=True)

   # ------------------------------------------

   # Initialize client A and client B's models
   client_shallow = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_medium = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_deep = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
                      
   clients = [client_shallow, client_medium, client_deep]
   training_loaders = [train_loader_shallow, train_loader_medium, train_loader_deep]


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

   # test 5 on the global server after normalized on 5/10/15
   _shallow_shallowCoojaRuns = [dataPreparation(df) for df in sequenced_shallow_Test_shallow]
   _medium_shallowCoojaRuns = [dataPreparation(df) for df in sequenced_medium_Test_shallow]
   _deep_shallowCoojaRuns = [dataPreparation(df) for df in sequenced_deep_Test_shallow]

   # test 10 on the global server after normalized on 5/10/15
   _shallow_mediumCoojaRuns = [dataPreparation(df) for df in sequenced_shallow_Test_medium]
   _medium_mediumCoojaRuns = [dataPreparation(df) for df in sequenced_medium_Test_medium]
   _deep_mediumCoojaRuns = [dataPreparation(df) for df in sequenced_deep_Test_medium]

   # test 15 on the global server after normalized on 5/10/15
   _shallow_deepCoojaRuns = [dataPreparation(df) for df in sequenced_shallow_Test_deep]
   _medium_deepCoojaRuns = [dataPreparation(df) for df in sequenced_medium_Test_deep]
   _deep_deepCoojaRuns = [dataPreparation(df) for df in sequenced_deep_Test_deep]

   
   

   # 
   lenshallow_shallowCoojaRuns = [len(dl.dataset) for dl in _shallow_shallowCoojaRuns]
   lenmedium_shallowCoojaRuns = [len(dl.dataset) for dl in _medium_shallowCoojaRuns]
   lendeep_shallowCoojaRuns = [len(dl.dataset) for dl in _deep_shallowCoojaRuns]

   # 
   lenshallow_mediumCoojaRuns = [len(dl.dataset) for dl in _shallow_mediumCoojaRuns]
   lenmedium_mediumCoojaRuns = [len(dl.dataset) for dl in _medium_mediumCoojaRuns]
   lendeep_mediumCoojaRuns = [len(dl.dataset) for dl in _deep_mediumCoojaRuns]

   # 
   lenshallow_deepCoojaRuns = [len(dl.dataset) for dl in _shallow_deepCoojaRuns]
   lenmedium_deepCoojaRuns = [len(dl.dataset) for dl in _medium_deepCoojaRuns]
   lendeep_deepCoojaRuns = [len(dl.dataset) for dl in _deep_deepCoojaRuns]



   #
   _shallow_shallow_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_shallow_shallowCoojaRuns, transition_indices_shallow_shallow)]
   _medium_shallow_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_medium_shallowCoojaRuns, transition_indices_medium_shallow)]
   _deep_shallow_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_deep_shallowCoojaRuns, transition_indices_deep_shallow)]

   _shallow_medium_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_shallow_mediumCoojaRuns, transition_indices_shallow_medium)]
   _medium_medium_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_medium_mediumCoojaRuns, transition_indices_medium_medium)]
   _deep_medium_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_deep_mediumCoojaRuns, transition_indices_deep_medium)]

   _shallow_deep_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_shallow_deepCoojaRuns, transition_indices_shallow_deep)]
   _medium_deep_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_medium_deepCoojaRuns, transition_indices_medium_deep)]
   _deep_deep_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_deep_deepCoojaRuns, transition_indices_deep_deep)]



   # bh_dis  means dis test that ius normalized by bh data             
   with open(add + '/detect_time_fed_shallow_shallow.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_shallow_shallow_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_medium_shallow.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_medium_shallow_CoojaRunPrediction)

   with open(add + '/detect_time_fed_deep_shallow.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_deep_shallow_CoojaRunPrediction)
   


   with open(add + '/detect_time_fed_shallow_medium.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_shallow_medium_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_medium_medium.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_medium_medium_CoojaRunPrediction)

   with open(add + '/detect_time_fed_deep_medium.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_deep_medium_CoojaRunPrediction)
   


   with open(add + '/detect_time_fed_shallow_deep.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_shallow_deep_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_medium_deep.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_medium_deep_CoojaRunPrediction)

   with open(add + '/detect_time_fed_deep_deep.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_deep_deep_CoojaRunPrediction)
   


