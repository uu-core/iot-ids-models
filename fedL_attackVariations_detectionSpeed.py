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
We test attack detection accuracy over time. The attacks that are tested are the base, on/off and gradual changing attack variations.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime (FL): attack variations (base, on/off, gradual changing)")

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
   _base_Train, _base_Test = aggregate.aggregate_list_base()
   _oo_Train, _oo_Test = aggregate.aggregate_list_oo()
   _dec_Train, _dec_Test = aggregate.aggregate_list_dec()



   ##   5
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _base_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __base_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __base_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __base_max = __base_all_maxs_df.max(axis = 0)
   __base_min = __base_all_mins_df.min(axis = 0)
   __base_normalized_dfs_train = [df.apply(lambda x: (x - __base_min[x.name]) / (__base_max[x.name] - __base_min[x.name])) for df in _base_Train]
   #__base_normalized_dfs_test = [df.apply(lambda x: (x - __base_min[x.name]) / (__base_max[x.name] - __base_min[x.name])) for df in _base_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   ##   10
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _oo_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __oo_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __oo_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __oo_max = __oo_all_maxs_df.max(axis = 0)
   __oo_min = __oo_all_mins_df.min(axis = 0)
   __oo_normalized_dfs_train = [df.apply(lambda x: (x - __oo_min[x.name]) / (__oo_max[x.name] - __oo_min[x.name])) for df in _oo_Train]
   #__oo_normalized_dfs_test = [df.apply(lambda x: (x - __oo_min[x.name]) / (__oo_max[x.name] - __oo_min[x.name])) for df in _oo_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   ##   15
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in _dec_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   __dec_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   __dec_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   __dec_max = __dec_all_maxs_df.max(axis = 0)
   __dec_min = __dec_all_mins_df.min(axis = 0)
   __dec_normalized_dfs_train = [df.apply(lambda x: (x - __dec_min[x.name]) / (__dec_max[x.name] - __dec_min[x.name])) for df in _dec_Train]
   #__dec_normalized_dfs_test = [df.apply(lambda x: (x - __dec_min[x.name]) / (__dec_max[x.name] - __dec_min[x.name])) for df in _dec_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   __base_normalized_dfs_test_base = [df.apply(lambda x: (x - __base_min[x.name]) / (__base_max[x.name] - __base_min[x.name])) for df in _base_Test]
   transition_indices_base_base = [df['label'].apply(lambda x: x == 1).idxmax() for df in __base_normalized_dfs_test_base]
   len_base_base = [df.shape[0] for df in __base_normalized_dfs_test_base]

   __base_normalized_dfs_test_oo = [df.apply(lambda x: (x - __base_min[x.name]) / (__base_max[x.name] - __base_min[x.name])) for df in _oo_Test]
   transition_indices_base_oo = [df['label'].apply(lambda x: x == 1).idxmax() for df in __base_normalized_dfs_test_oo]
   len_base_oo = [df.shape[0] for df in __base_normalized_dfs_test_oo]

   __base_normalized_dfs_test_dec = [df.apply(lambda x: (x - __base_min[x.name]) / (__base_max[x.name] - __base_min[x.name])) for df in _dec_Test]
   transition_indices_base_dec = [df['label'].apply(lambda x: x == 1).idxmax() for df in __base_normalized_dfs_test_dec]
   len_base_dec = [df.shape[0] for df in __base_normalized_dfs_test_dec]

   #

   __oo_normalized_dfs_test_base = [df.apply(lambda x: (x - __oo_min[x.name]) / (__oo_max[x.name] - __oo_min[x.name])) for df in _base_Test]    
   transition_indices_oo_base = [df['label'].apply(lambda x: x == 1).idxmax() for df in __oo_normalized_dfs_test_base]
   len_oo_base = [df.shape[0] for df in __oo_normalized_dfs_test_base]

   __oo_normalized_dfs_test_oo = [df.apply(lambda x: (x - __oo_min[x.name]) / (__oo_max[x.name] - __oo_min[x.name])) for df in _oo_Test]
   transition_indices_oo_oo = [df['label'].apply(lambda x: x == 1).idxmax() for df in __oo_normalized_dfs_test_oo]
   len_oo_oo = [df.shape[0] for df in __oo_normalized_dfs_test_oo]

   __oo_normalized_dfs_test_dec = [df.apply(lambda x: (x - __oo_min[x.name]) / (__oo_max[x.name] - __oo_min[x.name])) for df in _dec_Test]
   transition_indices_oo_dec = [df['label'].apply(lambda x: x == 1).idxmax() for df in __oo_normalized_dfs_test_dec]
   len_oo_dec = [df.shape[0] for df in __oo_normalized_dfs_test_dec]

   #

   __dec_normalized_dfs_test_base = [df.apply(lambda x: (x - __dec_min[x.name]) / (__dec_max[x.name] - __dec_min[x.name])) for df in _base_Test]
   transition_indices_dec_base = [df['label'].apply(lambda x: x == 1).idxmax() for df in __dec_normalized_dfs_test_base]
   len_dec_base = [df.shape[0] for df in __dec_normalized_dfs_test_base]

   __dec_normalized_dfs_test_oo = [df.apply(lambda x: (x - __dec_min[x.name]) / (__dec_max[x.name] - __dec_min[x.name])) for df in _oo_Test]
   transition_indices_dec_oo = [df['label'].apply(lambda x: x == 1).idxmax() for df in __dec_normalized_dfs_test_oo]
   len_dec_oo = [df.shape[0] for df in __dec_normalized_dfs_test_oo]

   __dec_normalized_dfs_test_dec = [df.apply(lambda x: (x - __dec_min[x.name]) / (__dec_max[x.name] - __dec_min[x.name])) for df in _dec_Test]
   transition_indices_dec_dec = [df['label'].apply(lambda x: x == 1).idxmax() for df in __dec_normalized_dfs_test_dec]
   len_dec_dec = [df.shape[0] for df in __dec_normalized_dfs_test_dec]





   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   
   sequenced_base_Train = [seqMaker.seq_maker(df,10) for df in __base_normalized_dfs_train]
   sequenced_base_Test_base = [seqMaker.seq_maker(df,10) for df in __base_normalized_dfs_test_base]
   sequenced_base_Test_oo = [seqMaker.seq_maker(df,10) for df in __base_normalized_dfs_test_oo]
   sequenced_base_Test_dec = [seqMaker.seq_maker(df,10) for df in __base_normalized_dfs_test_dec]

   sequenced_oo_Train = [seqMaker.seq_maker(df,10) for df in __oo_normalized_dfs_train]
   sequenced_oo_Test_base = [seqMaker.seq_maker(df,10) for df in __oo_normalized_dfs_test_base]
   sequenced_oo_Test_oo = [seqMaker.seq_maker(df,10) for df in __oo_normalized_dfs_test_oo]
   sequenced_oo_Test_dec = [seqMaker.seq_maker(df,10) for df in __oo_normalized_dfs_test_dec]

   sequenced_dec_Train = [seqMaker.seq_maker(df,10) for df in __dec_normalized_dfs_train]
   sequenced_dec_Test_base = [seqMaker.seq_maker(df,10) for df in __dec_normalized_dfs_test_base]
   sequenced_dec_Test_oo = [seqMaker.seq_maker(df,10) for df in __dec_normalized_dfs_test_oo]
   sequenced_dec_Test_dec = [seqMaker.seq_maker(df,10) for df in __dec_normalized_dfs_test_dec]


   # 
   sequenced_base_Train = pd.concat(sequenced_base_Train, ignore_index=True)
   sequenced_oo_Train = pd.concat(sequenced_oo_Train, ignore_index=True)
   sequenced_dec_Train = pd.concat(sequenced_dec_Train, ignore_index=True)
   
   
   # ------------------------------------------------------------------------------------------------------------ 

   # extract X and y 
   X_base_train = sequenced_base_Train.iloc[:, :-1].values  # All columns except the last one
   y_base_train = sequenced_base_Train.iloc[:, -1].values  # The last column
   X_oo_train = sequenced_oo_Train.iloc[:, :-1].values  # All columns except the last one
   y_oo_train = sequenced_oo_Train.iloc[:, -1].values  # The last column
   X_dec_train = sequenced_dec_Train.iloc[:, :-1].values  # All columns except the last one
   y_dec_train = sequenced_dec_Train.iloc[:, -1].values  # The last column
   

   # ----------------------------------------
   # tensorizing the data for 5
   X_base_train = np.array(X_base_train)
   y_base_train = np.array(y_base_train)

   X_base_train = torch.tensor(X_base_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_base_train = torch.tensor(y_base_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_base_train = torch.nan_to_num(X_base_train, nan=0.0)
   #y_base_train = torch.nan_to_num(y_base_train, nan=0.0)
   


   # tensorizing the data for 10
   X_oo_train = np.array(X_oo_train)
   y_oo_train = np.array(y_oo_train)
   
   X_oo_train = torch.tensor(X_oo_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_oo_train = torch.tensor(y_oo_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_oo_train = torch.nan_to_num(X_oo_train, nan=0.0)
   #y_base_train = torch.nan_to_num(y_base_train, nan=0.0)
   

   # tensorizing the data for 15
   X_dec_train = np.array(X_dec_train)
   y_dec_train = np.array(y_dec_train)

   X_dec_train = torch.tensor(X_dec_train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   y_dec_train = torch.tensor(y_dec_train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   X_dec_train = torch.nan_to_num(X_dec_train, nan=0.0)
   #y_base_train = torch.nan_to_num(y_base_train, nan=0.0)

   
   # ------------------------------------------------
   # Reshape your input to add sequence length dimension for 5
   X_base_train = X_base_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_base = TensorDataset(X_base_train, y_base_train)

   # Reshape your input to add sequence length dimension for 10
   X_oo_train = X_oo_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_oo = TensorDataset(X_oo_train, y_oo_train)

   # Reshape your input to add sequence length dimension for 15
   X_dec_train = X_dec_train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_dec = TensorDataset(X_dec_train, y_dec_train)

   # ------------------------------------------
   # Create DataLoaders for train and test sets 5
   train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 10
   train_loader_oo = DataLoader(train_dataset_oo, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets 15
   train_loader_dec = DataLoader(train_dataset_dec, batch_size=batch_size, shuffle=True)

   # ------------------------------------------

   # Initialize client A and client B's models
   client_base = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_oo = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_dec = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
                      
   clients = [client_base, client_oo, client_dec]
   training_loaders = [train_loader_base, train_loader_oo, train_loader_dec]


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
   _base_baseCoojaRuns = [dataPreparation(df) for df in sequenced_base_Test_base]
   _oo_baseCoojaRuns = [dataPreparation(df) for df in sequenced_oo_Test_base]
   _dec_baseCoojaRuns = [dataPreparation(df) for df in sequenced_dec_Test_base]

   # test 10 on the global server after normalized on 5/10/15
   _base_ooCoojaRuns = [dataPreparation(df) for df in sequenced_base_Test_oo]
   _oo_ooCoojaRuns = [dataPreparation(df) for df in sequenced_oo_Test_oo]
   _dec_ooCoojaRuns = [dataPreparation(df) for df in sequenced_dec_Test_oo]

   # test 15 on the global server after normalized on 5/10/15
   _base_decCoojaRuns = [dataPreparation(df) for df in sequenced_base_Test_dec]
   _oo_decCoojaRuns = [dataPreparation(df) for df in sequenced_oo_Test_dec]
   _dec_decCoojaRuns = [dataPreparation(df) for df in sequenced_dec_Test_dec]

   
   

   # 
   lenbase_baseCoojaRuns = [len(dl.dataset) for dl in _base_baseCoojaRuns]
   lenoo_baseCoojaRuns = [len(dl.dataset) for dl in _oo_baseCoojaRuns]
   lendec_baseCoojaRuns = [len(dl.dataset) for dl in _dec_baseCoojaRuns]

   # 
   lenbase_ooCoojaRuns = [len(dl.dataset) for dl in _base_ooCoojaRuns]
   lenoo_ooCoojaRuns = [len(dl.dataset) for dl in _oo_ooCoojaRuns]
   lendec_ooCoojaRuns = [len(dl.dataset) for dl in _dec_ooCoojaRuns]

   # 
   lenbase_decCoojaRuns = [len(dl.dataset) for dl in _base_decCoojaRuns]
   lenoo_decCoojaRuns = [len(dl.dataset) for dl in _oo_decCoojaRuns]
   lendec_decCoojaRuns = [len(dl.dataset) for dl in _dec_decCoojaRuns]



   #
   _base_base_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_base_baseCoojaRuns, transition_indices_base_base)]
   _oo_base_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_oo_baseCoojaRuns, transition_indices_oo_base)]
   _dec_base_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_dec_baseCoojaRuns, transition_indices_dec_base)]

   _base_oo_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_base_ooCoojaRuns, transition_indices_base_oo)]
   _oo_oo_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_oo_ooCoojaRuns, transition_indices_oo_oo)]
   _dec_oo_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_dec_ooCoojaRuns, transition_indices_dec_oo)]

   _base_dec_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_base_decCoojaRuns, transition_indices_base_dec)]
   _oo_dec_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_oo_decCoojaRuns, transition_indices_oo_dec)]
   _dec_dec_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_dec_decCoojaRuns, transition_indices_dec_dec)]



   # bh_dis  means dis test that ius normalized by bh data             
   with open(add + '/detect_time_fed_base_base.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_base_base_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_oo_base.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_oo_base_CoojaRunPrediction)

   with open(add + '/detect_time_fed_dec_base.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_dec_base_CoojaRunPrediction)
   


   with open(add + '/detect_time_fed_base_oo.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_base_oo_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_oo_oo.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_oo_oo_CoojaRunPrediction)

   with open(add + '/detect_time_fed_dec_oo.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_dec_oo_CoojaRunPrediction)
   


   with open(add + '/detect_time_fed_base_dec.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_base_dec_CoojaRunPrediction)
   
   with open(add + '/detect_time_fed_oo_dec.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_oo_dec_CoojaRunPrediction)

   with open(add + '/detect_time_fed_dec_dec.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_dec_dec_CoojaRunPrediction)
   


