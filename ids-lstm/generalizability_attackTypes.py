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
In this scenario we have 2 IDS models, one gets trained with the Blackhole attack type and the other one trained with the DIS-Flooding attack type.
We test the generalizability of each model to detect the other attack types.
We train and test each mudel --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Generalizability - Attack Types: ")

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




roc_auc_bh_bh = []             # records ROC-AUC of a model that is trained with the Blackhole attack data and tested with the Blackhole attack data
roc_auc_bh_dis = []            # records ROC-AUC of a model that is trained with the Blackhole attack data and tested with the DIS-Flooding attack data
roc_auc_dis_bh = []            # records ROC-AUC of a model that is trained with the DIS-Flooding attack data and tested with the Blackhole attack data
roc_auc_dis_dis = []           # records ROC-AUC of a model that is trained with the DIS-Flooding attack data and tested with the DIS-Flooding attack data






for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   bhTrain, bhTest = aggregate.aggregateBH_list()
   disTrain, disTest = aggregate.aggregateDIS_list()

   print(".............")
   print("Normalize bh: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".............")

   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in bhTrain]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   bh_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   bh_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

   bh_max = bh_all_maxs_df.max(axis = 0)
   bh_min = bh_all_mins_df.min(axis = 0)
   bh_normalized_dfs_train = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in bhTrain]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)


   print(".............")
   print("Normalize dis: ")
   # The attack data constitiutes of multiple scenarios implemented in the Cooja simulator. Therefore there are a min and a max of each feature in each scenario. 
   # To normalize the dataset we find the global min and max and then normalize the data with reapect to them
   print(".............")
   min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in disTrain]

   # Separate lists to store min and max DataFrames
   min_dfs = [min_df for min_df, max_df in min_max_list]
   max_dfs = [max_df for min_df, max_df in min_max_list]

   # Concatenate all min and max DataFrames into one DataFrame each
   dis_all_mins_df = pd.concat(min_dfs, ignore_index=True)
   dis_all_maxs_df = pd.concat(max_dfs, ignore_index=True)
   dis_max = dis_all_maxs_df.max(axis = 0)
   dis_min = dis_all_mins_df.min(axis = 0)
   dis_normalized_dfs_train = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in disTrain]

   del(min_max_list)
   del(min_dfs)
   del(max_dfs)



   # ------------------------------------
   # Normalize all test data
   # ------------------------------------
   bh_normalized_dfs_test_bh = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in bhTest]
   bh_normalized_dfs_test_dis = [df.apply(lambda x: (x - bh_min[x.name]) / (bh_max[x.name] - bh_min[x.name])) for df in disTest]

   dis_normalized_dfs_test_bh = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in bhTest]
   dis_normalized_dfs_test_dis = [df.apply(lambda x: (x - dis_min[x.name]) / (dis_max[x.name] - dis_min[x.name])) for df in disTest]

   # ------------------------------------
   # make them all seq and the concatenate
   sequencedbh_Train = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_train]
   sequencedbh_Test_bh = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_test_bh]
   sequencedbh_Test_dis = [seqMaker.seq_maker(df,10) for df in bh_normalized_dfs_test_dis]

   sequenceddis_Train = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_train]
   sequenceddis_Test_bh = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_test_bh]
   sequenceddis_Test_dis = [seqMaker.seq_maker(df,10) for df in dis_normalized_dfs_test_dis]

   sequencedbh_Train = pd.concat(sequencedbh_Train, ignore_index=True)
   sequencedbh_Test_bh = pd.concat(sequencedbh_Test_bh, ignore_index=True)
   sequencedbh_Test_dis = pd.concat(sequencedbh_Test_dis, ignore_index=True)

   sequenceddis_Train = pd.concat(sequenceddis_Train, ignore_index=True)
   sequenceddis_Test_bh = pd.concat(sequenceddis_Test_bh, ignore_index=True)
   sequenceddis_Test_dis = pd.concat(sequenceddis_Test_dis, ignore_index=True)


   


   # extract X and y for bh, dis
   Xbh_Train = sequencedbh_Train.iloc[:, :-1].values  # All columns except the last one
   ybh_Train = sequencedbh_Train.iloc[:, -1].values  # The last column
   Xdis_Train = sequenceddis_Train.iloc[:, :-1].values  # All columns except the last one
   ydis_Train = sequenceddis_Train.iloc[:, -1].values  # The last column
   
   Xbh_Test_bh = sequencedbh_Test_bh.iloc[:, :-1].values  # All columns except the last one
   ybh_Test_bh = sequencedbh_Test_bh.iloc[:, -1].values  # The last column
   Xbh_Test_dis = sequencedbh_Test_dis.iloc[:, :-1].values  # All columns except the last one
   ybh_Test_dis = sequencedbh_Test_dis.iloc[:, -1].values  # The last column
   
   Xdis_Test_bh = sequenceddis_Test_bh.iloc[:, :-1].values  # All columns except the last one
   ydis_Test_bh = sequenceddis_Test_bh.iloc[:, -1].values  # The last column
   Xdis_Test_dis = sequenceddis_Test_dis.iloc[:, :-1].values  # All columns except the last one
   ydis_Test_dis = sequenceddis_Test_dis.iloc[:, -1].values  # The last column
   

   # tensorizing the data for bh
   # ----------------------------------------
   Xbh_Train = np.array(Xbh_Train)
   ybh_Train = np.array(ybh_Train)
   Xbh_Test_bh = np.array(Xbh_Test_bh)
   ybh_Test_bh = np.array(ybh_Test_bh)
   Xbh_Test_dis = np.array(Xbh_Test_dis)
   ybh_Test_dis = np.array(ybh_Test_dis)

   Xbh_Train = torch.tensor(Xbh_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ybh_Train = torch.tensor(ybh_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xbh_Test_bh = torch.tensor(Xbh_Test_bh, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ybh_Test_bh = torch.tensor(ybh_Test_bh, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xbh_Test_dis = torch.tensor(Xbh_Test_dis, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ybh_Test_dis = torch.tensor(ybh_Test_dis, dtype = torch.long)     # Shape will be (num_samples, 1)

   # Check for NaNs due to min-max normalization
   Xbh_Train = torch.nan_to_num(Xbh_Train, nan=0.0)
   Xbh_Test_bh = torch.nan_to_num(Xbh_Test_bh, nan=0.0)
   Xbh_Test_dis = torch.nan_to_num(Xbh_Test_dis, nan=0.0)

   # tensorizing the data for dis
   # ----------------------------------------
   Xdis_Train = np.array(Xdis_Train)
   ydis_Train = np.array(ydis_Train)
   Xdis_Test_bh = np.array(Xdis_Test_bh)
   ydis_Test_bh = np.array(ydis_Test_bh)
   Xdis_Test_dis = np.array(Xdis_Test_dis)
   ydis_Test_dis = np.array(ydis_Test_dis)

   Xdis_Train = torch.tensor(Xdis_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   ydis_Train = torch.tensor(ydis_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xdis_Test_bh = torch.tensor(Xdis_Test_bh, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydis_Test_bh = torch.tensor(ydis_Test_bh, dtype = torch.long)     # Shape will be (num_samples, 1)
   Xdis_Test_dis = torch.tensor(Xdis_Test_dis, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   ydis_Test_dis = torch.tensor(ydis_Test_dis, dtype = torch.long)     # Shape will be (num_samples, 1)

   # Check for NaNs due to min-max normalization
   Xdis_Train = torch.nan_to_num(Xdis_Train, nan=0.0)
   Xdis_Test_bh = torch.nan_to_num(Xdis_Test_bh, nan=0.0)
   Xdis_Test_dis = torch.nan_to_num(Xdis_Test_dis, nan=0.0)
   

   # ----------------------------------------
   
   # Reshape your input to add sequence length dimension for BH
   Xbh_Train = Xbh_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xbh_Test_bh = Xbh_Test_bh.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xbh_Test_dis = Xbh_Test_dis.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_bh = TensorDataset(Xbh_Train, ybh_Train)
   test_dataset_bh_bh = TensorDataset(Xbh_Test_bh, ybh_Test_bh)
   test_dataset_bh_dis = TensorDataset(Xbh_Test_dis, ybh_Test_dis)

   # Reshape your input to add sequence length dimension for DIS
   Xdis_Train = Xdis_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdis_Test_bh = Xdis_Test_bh.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xdis_Test_dis = Xdis_Test_dis.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_dis = TensorDataset(Xdis_Train, ydis_Train)
   test_dataset_dis_bh = TensorDataset(Xdis_Test_bh, ydis_Test_bh)
   test_dataset_dis_dis = TensorDataset(Xdis_Test_dis, ydis_Test_dis)

   # Create DataLoaders for train and test sets - BH
   train_loader_bh = DataLoader(train_dataset_bh, batch_size=batch_size, shuffle=True)
   test_loader_bh_bh = DataLoader(test_dataset_bh_bh, batch_size=batch_size, shuffle=False)
   test_loader_bh_dis = DataLoader(test_dataset_bh_dis, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets - DIS
   train_loader_dis = DataLoader(train_dataset_dis, batch_size=batch_size, shuffle=True)
   test_loader_dis_bh = DataLoader(test_dataset_dis_bh, batch_size=batch_size, shuffle=False)
   test_loader_dis_dis = DataLoader(test_dataset_dis_dis, batch_size=batch_size, shuffle=False)

   # define two IDS models
   model_bh = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)
   model_dis = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

   
   # Train and Test BH
   pr.prGreen("Start Training!")
   model_bh.model_train(epochs = epochs,train_loader = train_loader_bh)
   pr.prGreen("__ BH Trained!")
   model_bh.check_model_nans(test_loader_bh_bh)
   model_bh.check_model_nans(test_loader_bh_dis)
   roc_auc_bh_bh.append(round(model_bh.evaluate_model_ROCAUC(test_loader_bh_bh),3))
   roc_auc_bh_dis.append(round(model_bh.evaluate_model_ROCAUC(test_loader_bh_dis),3))


   # Train and TestDIS
   pr.prGreen("Start Training!")
   model_dis.model_train(epochs = epochs,train_loader = train_loader_dis)
   pr.prGreen("__ DIS Trained!")
   model_dis.check_model_nans(test_loader_dis_bh)
   model_dis.check_model_nans(test_loader_dis_dis)
   roc_auc_dis_bh.append(round(model_dis.evaluate_model_ROCAUC(test_loader_dis_bh),3))
   roc_auc_dis_dis.append(round(model_dis.evaluate_model_ROCAUC(test_loader_dis_dis),3))





# save data
with open(add + '/roc_auc_bh_bh.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_bh_bh)

with open(add + '/roc_auc_bh_dis.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_bh_dis)



with open(add + '/roc_auc_dis_bh.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_dis_bh)

with open(add + '/roc_auc_dis_dis.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_dis_dis)

