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
We test the generalizability of the model to detect on networks with size 5, 10 , 15 and 20.
We train and test the model --runs-- times. The train and test data in each run is selected randomly.
"""



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description = "Global Model with data sharing - test on networks with different sizes")

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




roc_auc_all_all = []              # records ROC-AUC of a model that is trained with all data and tested with the all data
roc_auc_all_5 = []                # records ROC-AUC of a model that is trained with all data and tested on networks with size 5
roc_auc_all_10 = []               # records ROC-AUC of a model that is trained with all data and tested on networks with size 10
roc_auc_all_15 = []               # records ROC-AUC of a model that is trained with all data and tested on networks with size 15
roc_auc_all_20 = []               # records ROC-AUC of a model that is trained with all data and tested on networks with size 20






for run in range(runs):
   pr.prGreen("Run " + str(run))
   print(".................................")
   all_Train, all_Test = aggregate.aggregate_list_all()
   _5_Train, _5_Test = aggregate.aggregate_list_5()
   _10_Train, _10_Test = aggregate.aggregate_list_10()
   _15_Train, _15_Test = aggregate.aggregate_list_15()
   _20_Train, _20_Test = aggregate.aggregate_list_20()

   pr.prGreen(len(all_Train))
   pr.prGreen(len(all_Test))
   pr.prGreen(len(_5_Train))
   pr.prGreen(len(_5_Test))
   pr.prRed(len(_10_Train))
   pr.prRed(len(_10_Test))
   pr.prGreen(len(_15_Train))
   pr.prGreen(len(_15_Test))
   pr.prRed(len(_20_Train))
   pr.prRed(len(_20_Test))


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
   print("Normalize 5 : ")
   print(".................................")
   
   __5_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _5_Test]

   print(".................................")
   print("Normalize 10 : ")
   print(".................................")
   __10_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _10_Test]

   print(".................................")
   print("Normalize 15 : ")
   print(".................................")

   __15_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _15_Test]

   print(".................................")
   print("Normalize 20 : ")
   print(".................................")

   __20_normalized_dfs_test = [df.apply(lambda x: (x - all_min[x.name]) / (all_max[x.name] - all_min[x.name])) for df in _20_Test]


   # make them all seq and the concatenate
   sequencedall_Train = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_train]
   sequencedall_Test = [seqMaker.seq_maker(df,10) for df in all_normalized_dfs_test]
   sequenced_5_test = [seqMaker.seq_maker(df,10) for df in __5_normalized_dfs_test]
   sequenced_10_test = [seqMaker.seq_maker(df,10) for df in __10_normalized_dfs_test]
   sequenced_15_test = [seqMaker.seq_maker(df,10) for df in __15_normalized_dfs_test]
   sequenced_20_test = [seqMaker.seq_maker(df,10) for df in __20_normalized_dfs_test]


   sequencedall_Train = pd.concat(sequencedall_Train, ignore_index=True)
   sequencedall_Test = pd.concat(sequencedall_Test, ignore_index=True)
   sequenced_5_test = pd.concat(sequenced_5_test, ignore_index=True)
   sequenced_10_test = pd.concat(sequenced_10_test, ignore_index=True)
   sequenced_15_test = pd.concat(sequenced_15_test, ignore_index=True)
   sequenced_20_test = pd.concat(sequenced_20_test, ignore_index=True)


   # extract X and y for 5 to 20
   Xall_Train = sequencedall_Train.iloc[:, :-1].values  # All columns except the last one
   yall_Train = sequencedall_Train.iloc[:, -1].values  # The last column
   
   Xall_Test = sequencedall_Test.iloc[:, :-1].values  # All columns except the last one
   yall_Test = sequencedall_Test.iloc[:, -1].values  # The last column
   X_5_test = sequenced_5_test.iloc[:, :-1].values  # All columns except the last one
   y_5_test = sequenced_5_test.iloc[:, -1].values  # The last column
   X_10_test = sequenced_10_test.iloc[:, :-1].values  # All columns except the last one
   y_10_test = sequenced_10_test.iloc[:, -1].values  # The last column
   X_15_test = sequenced_15_test.iloc[:, :-1].values  # All columns except the last one
   y_15_test = sequenced_15_test.iloc[:, -1].values  # The last column
   X_20_test = sequenced_20_test.iloc[:, :-1].values  # All columns except the last one
   y_20_test = sequenced_20_test.iloc[:, -1].values  # The last column


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



   # tensorizing the data for 5
   X_5_test = np.array(X_5_test)
   y_5_test = np.array(y_5_test)
   X_5_test = torch.tensor(X_5_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_5_test = torch.tensor(y_5_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_5_test = torch.nan_to_num(X_5_test, nan=0.0)
   #y_5_test = torch.nan_to_num(y_5_test, nan=0.0)

   # tensorizing the data for 10
   X_10_test = np.array(X_10_test)
   y_10_test = np.array(y_10_test)
   X_10_test = torch.tensor(X_10_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_10_test = torch.tensor(y_10_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_10_test = torch.nan_to_num(X_10_test, nan=0.0)
   #y_10_test = torch.nan_to_num(y_10_test, nan=0.0)

   # tensorizing the data for 15
   X_15_test = np.array(X_15_test)
   y_15_test = np.array(y_15_test)
   X_15_test = torch.tensor(X_15_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_15_test = torch.tensor(y_15_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_15_test = torch.nan_to_num(X_15_test, nan=0.0)
   #y_15_test = torch.nan_to_num(y_15_test, nan=0.0)


   # tensorizing the data for 20
   X_20_test = np.array(X_20_test)
   y_20_test = np.array(y_20_test)
   X_20_test = torch.tensor(X_20_test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length, input_size)
   y_20_test = torch.tensor(y_20_test, dtype = torch.long)     # Shape will be (num_samples, 1)
   # Check for NaNs due to min-max normalization
   X_20_test = torch.nan_to_num(X_20_test, nan=0.0)
   #y_20_test = torch.nan_to_num(y_20_test, nan=0.0)


   Xall_Train = Xall_Train.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   Xall_Test = Xall_Test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   train_dataset_all = TensorDataset(Xall_Train, yall_Train)
   test_dataset_all = TensorDataset(Xall_Test, yall_Test)


   # Reshape your input to add sequence length dimension for 5
   X_5_test = X_5_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_5 = TensorDataset(X_5_test, y_5_test)

   # Reshape your input to add sequence length dimension for 10
   X_10_test = X_10_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_10 = TensorDataset(X_10_test, y_10_test)

   # Reshape your input to add sequence length dimension for 15
   X_15_test = X_15_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_15 = TensorDataset(X_15_test, y_15_test)

   # Reshape your input to add sequence length dimension for 20
   X_20_test = X_20_test.view(-1, 1, 140)  # Reshapes to [88301, 1, 140]
   test_dataset_20 = TensorDataset(X_20_test, y_20_test)


   # Create DataLoaders for train and test sets all
   train_loader_all = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=True)
   test_loader_all = DataLoader(test_dataset_all, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 5
   test_loader_5 = DataLoader(test_dataset_5, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 10
   test_loader_10 = DataLoader(test_dataset_10, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 15
   test_loader_15 = DataLoader(test_dataset_15, batch_size=batch_size, shuffle=False)

   # Create DataLoaders for train and test sets 20
   test_loader_20 = DataLoader(test_dataset_20, batch_size=batch_size, shuffle=False)



   # define the global model
   model_all = mine.LSTMClassifier(input_dim = 140, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = 2,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)



   # Train and Test all
   pr.prGreen("Start Training!")
   model_all.model_train(epochs = epochs,train_loader = train_loader_all)
   pr.prGreen("__ All Trained!")
   model_all.check_model_nans(test_loader_all)
   roc_auc_all_all.append(round(model_all.evaluate_model_ROCAUC(test_loader_all),3))
   # test on 5
   roc_auc_all_5.append(round(model_all.evaluate_model_ROCAUC(test_loader_5),3))
   # test on 10
   roc_auc_all_10.append(round(model_all.evaluate_model_ROCAUC(test_loader_10),3))
   # test on 15
   roc_auc_all_15.append(round(model_all.evaluate_model_ROCAUC(test_loader_15),3))
   # test on 20
   roc_auc_all_20.append(round(model_all.evaluate_model_ROCAUC(test_loader_20),3))
   


 
# save results
with open(add + '/roc_auc_all_all_12.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_all)

with open(add + '/roc_auc_all_5.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_5)

with open(add + '/roc_auc_all_10.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_10)

with open(add + '/roc_auc_all_15.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_15)

with open(add + '/roc_auc_all_20.csv', 'w',newline = '') as file:
   writer = csv.writer(file)
   writer.writerow(roc_auc_all_20)

