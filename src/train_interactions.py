from __future__ import absolute_import, print_function
import torch
import _init_paths

import os
import json
from opts import opts
from tracking_utils.utils import init_seeds
import numpy as np

import datasets.dataset as dataset
import models.own_blocks as models
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

def main(opt):
    init_seeds(opt.seed)
    #torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')

    train_dataset = dataset.InteractionTripletDataset(
        root = opt.interaction_data_root,
        file_name = opt.interaction_data_file,
        )
    
    val_dataset = dataset.InteractionTripletDataset(
        root=opt.interaction_data_root,
        file_name=opt.interaction_data_file.replace("train", "test"),
    )
    
    device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print(device)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    print('Creating model...')
    model = models.InteractionDetection()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    model.to(device)

    # Define the number of epochs
    num_epochs = opt.num_epochs

    # Training loop
    for epoch in range(num_epochs):
        # Reset running loss and correct predictions for each epoch
        train_running_loss = 0.0
        train_correct_predictions = 0
        
        model.train()
        
        for i, data in enumerate(train_dataloader):
            if i % 100 == 0:
                print(i)
            subj, obj, union, labels = data['subj'].to(device), data['obj'].to(device), data['union'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(subj=subj, obj=obj, union=union)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update the running loss
            train_running_loss += loss.item()
            
            # Calculate the predicted labels
            _, predicted_labels = torch.max(outputs, 1)
            
            # Count the correct predictions
            train_correct_predictions += (predicted_labels == labels).sum().item()
        
        # Calculate the average loss and accuracy for the training dataset
        train_average_loss = train_running_loss / len(train_dataloader)
        train_accuracy = train_correct_predictions / len(train_dataset)
        
        # Initialize variables for running loss and correct predictions
        val_running_loss = 0.0
        val_correct_predictions = 0

        with torch.no_grad():
            model.eval()
            # Iterate over the validation dataloader
            for data in val_dataloader:
                # Move the data to the device
                subj, obj, union, labels = data['subj'].to(device), data['obj'].to(device), data['union'].to(device), data['label'].to(device)
                
                # Forward pass
                outputs = model(subj=subj, obj=obj, union=union)
                
                # Calculate the loss
                loss = criterion(outputs, labels)
                
                # Update the running loss
                val_running_loss += loss.item()
                
                # Calculate the predicted labels
                _, predicted_labels = torch.max(outputs, 1)
                
                # Count the correct predictions
                val_correct_predictions += (predicted_labels == labels).sum().item()

        # Calculate the average loss and accuracy for the validation dataset
        val_average_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct_predictions / len(val_dataset)

        # Print the average loss and accuracy for the epoch
        print(f"Epoch {epoch+1} - Training Loss: {train_average_loss}, Training Accuracy: {train_accuracy}")
        print(f"Epoch {epoch+1} - Validation Loss: {val_average_loss}, Validation Accuracy: {val_accuracy}")  


        if not os.path.exists(opt.interaction_output_folder):
            os.makedirs(opt.interaction_output_folder)
        # Save the model after each epoch
        if opt.save_all and (epoch+1) % opt.val_intervals == 0:
            torch.save(model.state_dict(), f"{opt.interaction_output_folder}/model_epoch_{epoch+1}.pth")
        
        # Write the Training and Validation accuracy to the progress file
        with open(f"{opt.interaction_output_folder}/progress_file.txt", "a") as file:
            file.write(f"Epoch {epoch+1} - Training Loss: {train_average_loss}, Validation Loss: {val_average_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}\n")



if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
