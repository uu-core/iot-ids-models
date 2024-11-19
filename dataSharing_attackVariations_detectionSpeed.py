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
We test attack detection accuracy over time. The attacks that are tested are the base, on/off and gradual changing attack variations.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""




# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime: attack variations (base, on/off, gradual changing)")

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
   base_Train, base_Test = aggregate.aggregate_list_base()
   oo_Train, oo_Test = aggregate.aggregate_list_oo()
   dec_Train, dec_Test = aggregate.aggregate_list_dec()

   all_Train = base_Train + oo_Train + dec_Train
   all_Test = base_Test + oo_Test + dec_Test
   

   # pr.prGreen(len(all_Train))
   # pr.prGreen(len(all_Test))
   # pr.prGreen(len(_base_Train))
   # pr.prGreen(len(_base_Test))
   # pr.prRed(len(_oo_Train))
   # pr.prRed(len(_oo_Test))
   # pr.prGreen(len(_dec_Train))
   # pr.prGreen(len(_dec_Test))
   

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

   # Normalize Base test
   base_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in base_Test]
   transition_indices_base = [df['label'].apply(lambda x: x == 1).idxmax() for df in base_normalized_dfs_test]
   len_base = [df.shape[0] for df in base_normalized_dfs_test]
   
   # Normalize OO test
   oo_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in oo_Test]
   transition_indices_oo = [df['label'].apply(lambda x: x == 1).idxmax() for df in oo_normalized_dfs_test]
   len_oo = [df.shape[0] for df in oo_normalized_dfs_test]

   # Normalize DEC test
   dec_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in dec_Test]
   transition_indices_dec = [df['label'].apply(lambda x: x == 1).idxmax() for df in dec_normalized_dfs_test]
   len_dec = [df.shape[0] for df in dec_normalized_dfs_test]


   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   sequencedall_Train = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_train]
   sequencedall_Test = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_test]
   sequencedbase_Test = [seqMaker.seq_maker(df,10) for df in base_normalized_dfs_test]
   sequencedoo_Test = [seqMaker.seq_maker(df,10) for df in oo_normalized_dfs_test]
   sequenceddec_Test = [seqMaker.seq_maker(df,10) for df in dec_normalized_dfs_test]

   sequencedall_Train = pd.concat(sequencedall_Train, ignore_index=True)


   # ------------------------------------------------------------------------------------------------------------ 
   # extract X and y for all, bh, dis
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

      # Create DataLoaders for test sets bh
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

   baseCoojaRuns = [dataPreparation(df) for df in sequencedbase_Test]
   ooCoojaRuns = [dataPreparation(df) for df in sequencedoo_Test]
   decCoojaRuns = [dataPreparation(df) for df in sequenceddec_Test]

   lenBASECoojaRuns = [len(dl.dataset) for dl in baseCoojaRuns]
   lenOOCoojaRuns = [len(dl.dataset) for dl in ooCoojaRuns]
   lenDECCoojaRuns = [len(dl.dataset) for dl in decCoojaRuns]

   
   baseCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(baseCoojaRuns, transition_indices_base)]
   ooCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(ooCoojaRuns,transition_indices_oo)]
   decCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(decCoojaRuns,transition_indices_dec)]



   with open(add + '/detect_time_all_base.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(baseCoojaRunPrediction)

   with open(add + '/detect_time_all_oo.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(ooCoojaRunPrediction)

   with open(add + '/detect_time_all_dec.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(decCoojaRunPrediction)


   


