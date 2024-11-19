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






"""
In this scenario we train a global IDS model using the data sharing method.
We test attack detection accuracy over time. The evaluation is over data from networks with size 5, 10, 15 and 20.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""




# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime: network size (5, 10, 15, 20)")

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





for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   _5_Train, _5_Test = aggregate.aggregate_list_5()
   _10_Train, _10_Test = aggregate.aggregate_list_10()
   _15_Train, _15_Test = aggregate.aggregate_list_15()
   _20_Train, _20_Test = aggregate.aggregate_list_20()

   all_Train = _5_Train + _10_Train + _15_Train +_20_Train
   all_Test = _5_Test + _10_Test + _15_Test + _20_Test

   # pr.prGreen(len(all_Train))
   # pr.prGreen(len(all_Test))
   # pr.prGreen(len(_5_Train))
   # pr.prGreen(len(_5_Test))
   # pr.prRed(len(_10_Train))
   # pr.prRed(len(_10_Test))
   # pr.prGreen(len(_15_Train))
   # pr.prGreen(len(_15_Test))
   # pr.prRed(len(_20_Train))
   # pr.prRed(len(_20_Test))


   print("................................")
   print("Normalize all: ")
   print(".................................")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in all_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   all_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   all_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   all_max = all_all_maxs_df.max(axis = 0)
   all_min = all_all_mins_df.min(axis = 0)
   all_normalized_dfs_train = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in all_Train]
   all_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in all_Test]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)

   # Normalize 5 test
   _5_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _5_Test]
   transition_indices_5 = [df['label'].apply(lambda x: x == 1).idxmax() for df in _5_normalized_dfs_test]
   len_5 = [df.shape[0] for df in _5_normalized_dfs_test]

   # Normalize 10 test
   _10_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _10_Test]
   transition_indices_10 = [df['label'].apply(lambda x: x == 1).idxmax() for df in _10_normalized_dfs_test]
   len_10 = [df.shape[0] for df in _10_normalized_dfs_test]
   
   # Normalize 15 test
   _15_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _15_Test]
   transition_indices_15 = [df['label'].apply(lambda x: x == 1).idxmax() for df in _15_normalized_dfs_test]
   len_15 = [df.shape[0] for df in _15_normalized_dfs_test]

   # Normalize 20 test
   _20_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _20_Test]
   transition_indices_20 = [df['label'].apply(lambda x: x == 1).idxmax() for df in _20_normalized_dfs_test]
   len_20 = [df.shape[0] for df in _20_normalized_dfs_test]


   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   sequencedall_Train = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_train]
   sequencedall_Test = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_test]
   sequenced_5_Test = [seqMaker.seq_maker(df,10) for df in _5_normalized_dfs_test]
   sequenced_10_Test = [seqMaker.seq_maker(df,10) for df in _10_normalized_dfs_test]
   sequenced_15_Test = [seqMaker.seq_maker(df,10) for df in _15_normalized_dfs_test]
   sequenced_20_Test = [seqMaker.seq_maker(df,10) for df in _20_normalized_dfs_test]

   sequencedall_Train = pd.concat(sequencedall_Train, ignore_index=True)




   # ------------------------------------------------------------------------------------------------------------ 
   # extract X and y for all
   Xall_Train = sequencedall_Train.iloc[:, :-1].values  # All columns except the last one
   yall_Train = sequencedall_Train.iloc[:, -1].values  # The last column

   # ------------------------------------------------------------------------------------------------------------

   # tensorizing the data for all
   Xall_Train = np.array(Xall_Train)
   yall_Train = np.array(yall_Train)
   Xall_Train = torch.tensor(Xall_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   yall_Train = torch.tensor(yall_Train, dtype = torch.long)   # Shape will be (num_samples, 1)

   # Check for NaNs due to min-max normalization
   Xall_Train = torch.nan_to_num(Xall_Train, nan=0.0)

   # ------------------------------------------------------------------------------------------------------------

   # Reshape your input to add sequence length dimension for all
   Xall_Train = Xall_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_all = TensorDataset(Xall_Train, yall_Train)

   # ------------------------------------------------------------------------------------------------------------

   # Create DataLoaders for train and test sets all
   train_loader_all = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=True)

   # ------------------------------------------------------------------------------------------------------------


   model_all = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

   # Train and Test all
   pr.prGreen("Start Training!")
   model_all.model_train(epochs = epochs,train_loader = train_loader_all)
   pr.prGreen("__ all Trained!")


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

      # Create DataLoaders for test sets h
      myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=False)


      return myDataLoader


   def timeFinder(dl, trans):
      #print(".....................................................")
      for inputs, label in dl:
         detection = torch.argmax(model_all(inputs), dim=1)
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

   _5_CoojaRuns = [dataPreparation(df) for df in sequenced_5_Test]
   _10_CoojaRuns = [dataPreparation(df) for df in sequenced_10_Test]
   _15_CoojaRuns = [dataPreparation(df) for df in sequenced_15_Test]
   _20_CoojaRuns = [dataPreparation(df) for df in sequenced_20_Test]

   len5CoojaRuns = [len(dl.dataset) for dl in _5_CoojaRuns]
   len10CoojaRuns = [len(dl.dataset) for dl in _10_CoojaRuns]
   len15CoojaRuns = [len(dl.dataset) for dl in _15_CoojaRuns]
   len20CoojaRuns = [len(dl.dataset) for dl in _20_CoojaRuns]

   
   _5_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_5_CoojaRuns, transition_indices_5)]
   _10_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_10_CoojaRuns, transition_indices_10)]
   _15_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_15_CoojaRuns, transition_indices_15)]
   _20_CoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(_20_CoojaRuns, transition_indices_20)]
   


   with open(add + '/detect_time_all_5.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_5_CoojaRunPrediction)

   with open(add + '/detect_time_all_10.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_10_CoojaRunPrediction)

   with open(add + '/detect_time_all_15.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_15_CoojaRunPrediction)

   with open(add + '/detect_time_all_20.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(_20_CoojaRunPrediction)
