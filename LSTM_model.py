#!/usr/bin/env python3


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import roc_auc_score, roc_curve, auc
import coloredPrinting as pr






class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, fc_hidden_dim, learning_rate):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_train = []
        self.loss_test = []


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)  # Activation function for the hidden layer
        out = self.fc2(out)
        out = torch.softmax(out,dim = 1)

        return out


    def train_model_eval(self, epochs, training_data_fcn, test_x, test_y):
        # Training the model without dropout
        for epoch in range(epochs):
            for x_batch, y_batch in training_data_fcn:
                self.train()
                y_prediction = self(x_batch)
                loss = self.criterion(y_prediction,y_batch)
                
                self.optimizer.zero_grad()
                self.loss_train.append(loss.item())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Clip gradients
                self.optimizer.step()
                
                self.eval()
                with torch.no_grad():
                    output_test = self(test_x)
                    loss_t = self.criterion(output_test, test_y)
                    self.loss_test.append(loss_t.item())


    def train_model(self, epochs, training_data_fcn):
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for x_batch, y_batch in training_data_fcn:
                y_prediction = self(x_batch)
                loss = self.criterion(y_prediction,y_batch)
                
                self.optimizer.zero_grad()
                self.loss_train.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    def model_train(self, epochs, train_loader):
        for epoch in range(epochs):
            total_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_loader):
                # clear the gradients 
                self.optimizer.zero_grad()

                # forward pass
                y_prediction = self(x_batch)
                loss = self.criterion(y_prediction,y_batch)
                
                # backward pass and optimization
                self.loss_train.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            


    def eval_model(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            
        return outputs
    

    def check_model_nans(self, loader):
        for inputs, targets in loader:
            outputs = self(inputs)
            if torch.isnan(outputs).any():
                pr.prRed("NaNs detected in outputs")

    
    def evaluate_model_ROCAUC(self, test_loader):
        # Set the model to evaluation mode
        self.eval()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            
            for inputs, targets in test_loader:
                outputs = self(inputs)
                all_outputs.append(outputs)
                all_targets.append(targets)
            
            # Concatenate all batches to get complete set
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Convert outputs to probabilities using softmax and get probabilities for class 1
            #probabilities = torch.softmax(all_outputs, dim=1)[:, 1]
            probabilities = all_outputs[:,1]
            #all_targets = all_targets[:,1]
            
            # Calculate ROC AUC score
            #roc_auc = roc_auc_score(all_targets.cpu().numpy(), probabilities.cpu().numpy())
            probabilities = probabilities.detach().numpy()
            all_targets = all_targets.detach().numpy()
            model_fpr, model_tpr, threshold = roc_curve(all_targets, probabilities)
            roc_auc = auc(model_fpr, model_tpr)
            
        
        return roc_auc



