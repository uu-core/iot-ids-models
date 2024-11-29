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
In this scenario we have 1 IDS models, trained with all data.
We test the generalizability of the model to detect the base, the on/off and gradual changing attack variations.
We train and test the model --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Global Model with data sharing - test on the attack variations")

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

roc_auc_all_all = []             # records ROC-AUC of a model that is trained with all data and tested with the all data
roc_auc_all_base = []            # records ROC-AUC of a model that is trained with all data and tested with the base attack data
roc_auc_all_oo = []              # records ROC-AUC of a model that is trained with all data and tested with the on/off attack data
roc_auc_all_dec = []             # records ROC-AUC of a model that is trained with all data and tested with the gradual changing attack data







for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   all_Train, all_Test = aggregate.aggregate_list_all()
   _base_Train, _base_Test = aggregate.aggregate_list_base()
   _oo_Train, _oo_Test = aggregate.aggregate_list_oo()
   _dec_Train, _dec_Test = aggregate.aggregate_list_dec()
   

   pr.prGreen(len(all_Train))
   pr.prGreen(len(all_Test))
   pr.prGreen(len(_base_Train))
   pr.prGreen(len(_base_Test))
   pr.prRed(len(_oo_Train))
   pr.prRed(len(_oo_Test))
   pr.prGreen(len(_dec_Train))
   pr.prGreen(len(_dec_Test))
   

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




   print(".................................")
   print("Normalize BASE : ")
   print(".................................")
   
   __base_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _base_Test]

   print(".................................")
   print("Normalize OO : ")
   print(".................................")
   __oo_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _oo_Test]

   print(".................................")
   print("Normalize DEC : ")
   print(".................................")

   __dec_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _dec_Test]

   


   # make them all seq and the concatenate
   sequencedall_Train = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_train]
   sequencedall_Test = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_test]
   sequenced_base_test = [seqMaker.seq_maker(df,10) for df in __base_normalized_dfs_test]
   sequenced_oo_test = [seqMaker.seq_maker(df,10) for df in __oo_normalized_dfs_test]
   sequenced_dec_test = [seqMaker.seq_maker(df,10) for df in __dec_normalized_dfs_test]
   


   sequencedall_Train = pd.concat(sequencedall_Train, ignore_index=True)
   sequencedall_Test = pd.concat(sequencedall_Test, ignore_index=True)
   sequenced_base_test = pd.concat(sequenced_base_test, ignore_index=True)
   sequenced_oo_test = pd.concat(sequenced_oo_test, ignore_index=True)
   sequenced_dec_test = pd.concat(sequenced_dec_test, ignore_index=True)
   


   # extract X and y for base, oo and dec
   Xall_Train = sequencedall_Train.iloc[:, :-1].values  # All columns except the last one
   yall_Train = sequencedall_Train.iloc[:, -1].values  # The last column
   
   Xall_Test = sequencedall_Test.iloc[:, :-1].values  # All columns except the last one
   yall_Test = sequencedall_Test.iloc[:, -1].values  # The last column
   X_base_test = sequenced_base_test.iloc[:, :-1].values  # All columns except the last one
   y_base_test = sequenced_base_test.iloc[:, -1].values  # The last column
   X_oo_test = sequenced_oo_test.iloc[:, :-1].values  # All columns except the last one
   y_oo_test = sequenced_oo_test.iloc[:, -1].values  # The last column
   X_dec_test = sequenced_dec_test.iloc[:, :-1].values  # All columns except the last one
   y_dec_test = sequenced_dec_test.iloc[:, -1].values  # The last column
   

   # tensorizing the data for all
   Xall_Train = np.array(Xall_Train)
   yall_Train = np.array(yall_Train)
   Xall_Test = np.array(Xall_Test)
   yall_Test = np.array(yall_Test)
   Xall_Train = torch.tensor(Xall_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length, input_size)
   yall_Train = torch.tensor(yall_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
   Xall_Test = torch.tensor(Xall_Test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   yall_Test = torch.tensor(yall_Test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   Xall_Train = torch.nan_to_num(Xall_Train, nan=0.0)
   #yall_Train = torch.nan_to_num(yall_Train, nan=0.0)
   Xall_Test = torch.nan_to_num(Xall_Test, nan=0.0)
   #yall_Test = torch.nan_to_num(yall_Test, nan=0.0)



   # tensorizing the data for base
   X_base_test = np.array(X_base_test)
   y_base_test = np.array(y_base_test)
   X_base_test = torch.tensor(X_base_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_base_test = torch.tensor(y_base_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_base_test = torch.nan_to_num(X_base_test, nan=0.0)
   #y_base_test = torch.nan_to_num(y_base_test, nan=0.0)

   # tensorizing the data for OO
   X_oo_test = np.array(X_oo_test)
   y_oo_test = np.array(y_oo_test)
   X_oo_test = torch.tensor(X_oo_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_oo_test = torch.tensor(y_oo_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_oo_test = torch.nan_to_num(X_oo_test, nan=0.0)
   #y_oo_test = torch.nan_to_num(y_oo_test, nan=0.0)

   # tensorizing the data for dec
   X_dec_test = np.array(X_dec_test)
   y_dec_test = np.array(y_dec_test)
   X_dec_test = torch.tensor(X_dec_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_dec_test = torch.tensor(y_dec_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_dec_test = torch.nan_to_num(X_dec_test, nan=0.0)
   #y_dec_test = torch.nan_to_num(y_dec_test, nan=0.0)




   Xall_Train = Xall_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xall_Test = Xall_Test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_all = TensorDataset(Xall_Train, yall_Train)
   test_dataset_all = TensorDataset(Xall_Test, yall_Test)


   # Reshape your input to add sequence length dimension for base
   X_base_test = X_base_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_base = TensorDataset(X_base_test, y_base_test)

   # Reshape your input to add sequence length dimension for OO
   X_oo_test = X_oo_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_oo = TensorDataset(X_oo_test, y_oo_test)

   # Reshape your input to add sequence length dimension for dec
   X_dec_test = X_dec_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_dec = TensorDataset(X_dec_test, y_dec_test)

   

   # Create DataLoaders for train and test sets all
   train_loader_all = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=True)
   test_loader_all = DataLoader(test_dataset_all, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets base
   test_loader_base = DataLoader(test_dataset_base, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets OO
   test_loader_oo = DataLoader(test_dataset_oo, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets dec
   test_loader_dec = DataLoader(test_dataset_dec, batch_size=batch_size, shuffle=False)

   


   # define the global model
   model_all = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)



   # Train and Test all
   pr.prGreen("Start Training!")
   model_all.model_train(epochs = epochs,train_loader = train_loader_all)
   pr.prGreen("__ All Trained!")
   model_all.check_model_nans(test_loader_all)
   roc_auc_all_all.append(round(model_all.evaluate_model_ROCAUC(test_loader_all),3))
   # test on the base attck data 
   roc_auc_all_base.append(round(model_all.evaluate_model_ROCAUC(test_loader_base),3))
   # test on the on/off attck data 
   roc_auc_all_oo.append(round(model_all.evaluate_model_ROCAUC(test_loader_oo),3))
   # test on the gradual changing attck data 
   roc_auc_all_dec.append(round(model_all.evaluate_model_ROCAUC(test_loader_dec),3))
   
   


 
# save results
with open(add + '/roc_auc_all_all_13.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_all)

with open(add + '/roc_auc_all_base.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_base)

with open(add + '/roc_auc_all_oo.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_oo)

with open(add + '/roc_auc_all_dec.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_dec)


