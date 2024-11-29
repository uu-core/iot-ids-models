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
We test attack detection accuracy over time. The attacks that are tested are the BH and the DIS attack types.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Detection accuracy over tiime (FL): attack types (BH, DIS)")

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
   bh_Train, bh_Test = aggregate.aggregateBH_list()
   dis_Train, dis_Test = aggregate.aggregateDIS_list()



   print(".............")
   print("Normalize bh: ")
   print(".............")

   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in bh_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   bh_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   bh_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   bh_max = bh_all_maxs_df.max(axis = 0)
   bh_min = bh_all_mins_df.min(axis = 0)
   bh_normalized_dfs_train = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in bh_Train]
   # bh_normalized_dfs_test = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in bhTest]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".............")
   print("Normalize dis: ")
   print(".............")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in dis_Train]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   dis_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   dis_all_maxs_df = pd.concat(max_dfs, ignore_index=True)
   dis_max = dis_all_maxs_df.max(axis = 0)
   dis_min = dis_all_mins_df.min(axis = 0)
   dis_normalized_dfs_train = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in dis_Train]
   # dis_normalized_dfs_test = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in disTest]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)



   # Normalize BH test after normalized with (bh server)/(dis server)
   bh_normalized_dfs_test_bh = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in bh_Test]
   transition_indices_bh_bh = [df['label'].apply(lambda x: x == 1).idxmax() for df in bh_normalized_dfs_test_bh]
   len_bh_bh = [df.shape[0] for df in bh_normalized_dfs_test_bh]

   dis_normalized_dfs_test_bh = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in bh_Test]
   transition_indices_dis_bh = [df['label'].apply(lambda x: x == 1).idxmax() for df in dis_normalized_dfs_test_bh]
   len_dis_bh = [df.shape[0] for df in dis_normalized_dfs_test_bh]
   
   
   # Normalize DIS test
   bh_normalized_dfs_test_dis = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in dis_Test]
   transition_indices_bh_dis = [df['label'].apply(lambda x: x == 1).idxmax() for df in bh_normalized_dfs_test_dis]
   len_bh_bh = [df.shape[0] for df in bh_normalized_dfs_test_dis]
   
   
   dis_normalized_dfs_test_dis = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in dis_Test]
   transition_indices_dis_dis = [df['label'].apply(lambda x: x == 1).idxmax() for df in dis_normalized_dfs_test_dis]
   len_dis_dis = [df.shape[0] for df in dis_normalized_dfs_test_dis]


   # ------------------------------------------------------------------------------------------------------------
   # make them all seq and then concatenate
   
   sequencedbh_Train = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_train]
   sequencedbh_Test_bh = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_test_bh]
   sequencedbh_Test_dis = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_test_dis]

   sequenceddis_Train = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_train]
   sequenceddis_Test_bh = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_test_bh]
   sequenceddis_Test_dis = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_test_dis]

   sequencedbh_Train = pd.concat(sequencedbh_Train, ignore_index=True)
   sequenceddis_Train = pd.concat(sequenceddis_Train, ignore_index=True)
   
   # ------------------------------------------------------------------------------------------------------------ 

   # extract X and y for bh, dis
   Xbh_Train = sequencedbh_Train.iloc[:, :-1].values  # All columns except the last one
   ybh_Train = sequencedbh_Train.iloc[:, -1].values  # The last column
   Xdis_Train = sequenceddis_Train.iloc[:, :-1].values  # All columns except the last one
   ydis_Train = sequenceddis_Train.iloc[:, -1].values  # The last column
   
   # tensorizing the data for bh
   # ----------------------------------------
   Xbh_Train = np.array(Xbh_Train)
   ybh_Train = np.array(ybh_Train)
   
   Xbh_Train = torch.tensor(Xbh_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ybh_Train = torch.tensor(ybh_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   Xbh_Train = torch.nan_to_num(Xbh_Train, nan=0.0)
   #ybh_Train = torch.nan_to_num(ybh_Train, nan=0.0)
   

   # tensorizing the data for dis
   # ----------------------------------------
   Xdis_Train = np.array(Xdis_Train)
   ydis_Train = np.array(ydis_Train)
   
   Xdis_Train = torch.tensor(Xdis_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ydis_Train = torch.tensor(ydis_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   
   # Check for NaNs due to min-max normalization
   Xdis_Train = torch.nan_to_num(Xdis_Train, nan=0.0)
   #ybh_Train = torch.nan_to_num(ybh_Train, nan=0.0)
   
   # ----------------------------------------
   
   # Reshape your input to add sequence length dimension for bh
   Xbh_Train = Xbh_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_bh = TensorDataset(Xbh_Train, ybh_Train)
   
   # Reshape your input to add sequence length dimension for dis
   Xdis_Train = Xdis_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_dis = TensorDataset(Xdis_Train, ydis_Train)
   
   # Create DataLoaders for train and test sets bh
   train_loader_bh = DataLoader(train_dataset_bh, batch_size=batch_size, shuffle=True)

   # Create DataLoaders for train and test sets dis
   train_loader_dis = DataLoader(train_dataset_dis, batch_size=batch_size, shuffle=True)

   # Initialize client A and client B's models
   client_bh = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
   client_dis = LSTM_FED.LSTMModel(input_dim = 140, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers, fc_hidden_dim = fc_hidden_dim, learning_rate = lr, device = device)
                      

   clients = [client_bh, client_dis]
   training_loaders = [train_loader_bh, train_loader_dis]


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

   # test bh on the global server after normalized on bh/dis
   bh_bhCoojaRuns = [dataPreparation(df) for df in sequencedbh_Test_bh]
   dis_bhCoojaRuns = [dataPreparation(df) for df in sequenceddis_Test_bh]

   # test dis on the global server after normalized on bh/dis
   bh_disCoojaRuns = [dataPreparation(df) for df in sequencedbh_Test_dis]
   dis_disCoojaRuns = [dataPreparation(df) for df in sequenceddis_Test_dis]
   

   # 
   lenBH_BHCoojaRuns = [len(dl.dataset) for dl in bh_bhCoojaRuns]
   lenDIS_BHCoojaRuns = [len(dl.dataset) for dl in dis_bhCoojaRuns]

   lenBH_DISCoojaRuns = [len(dl.dataset) for dl in bh_disCoojaRuns]
   lenDIS_DISCoojaRuns = [len(dl.dataset) for dl in dis_disCoojaRuns]

   #
   bhbhCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(bh_bhCoojaRuns, transition_indices_bh_bh)]
   disbhCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(dis_bhCoojaRuns, transition_indices_dis_bh)]

   bhdisCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(bh_disCoojaRuns, transition_indices_bh_dis)]
   disdisCoojaRunPrediction = [timeFinder(dl,trans) for dl, trans in zip(dis_disCoojaRuns, transition_indices_dis_dis)]



   # bh_dis  means dis test that ius normalized by bh data             
   with open(add + '/detect_time_fed_bh_bh.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(bhbhCoojaRunPrediction)

   with open(add + '/detect_time_fed_dis_bh.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(disbhCoojaRunPrediction)

   with open(add + '/detect_time_fed_bh_dis.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(bhdisCoojaRunPrediction)

   with open(add + '/detect_time_fed_dis_dis.csv', 'w',newline = '') as file:
      writer = csv.writer(file)
      writer.writerow(disdisCoojaRunPrediction)

