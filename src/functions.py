import os
import shutil
import numpy as np
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# function 1: accuracy
def accuracy(predictions, labels, treshold)->int:
    '''
    This is a function for calculating the accuracy of a categorization model.
    
    predictions(array of floats): predictions of the model (outputs)
    labels(array of floats): binary data labels
    treshold(float): a value above which an event is triggered eg. a galaxy merger is shown 
    '''
    
    # converting the model's predictions into 0 or 1 using the threshold value
    # if the prediction is greater than the threshold, then it is 1, and if less then it is 0
    preds_b = (predictions > treshold).float()

    # summing all the predictions with the same labels (correctly categorized images)
    return (preds_b == labels).sum().item()

# function 2: training and validation loop
def training_loop(n_epochs, model, train_loader, valid_loader, loss_fn, optimizer, save_name:str=None):
    '''
    This is a function for training and validation of a model. 
    
    n_epochs(integer): number of epochs
    model(class): architecture of neural network
    train_loader(DataLoader class): training data
    valid_loader(DataLoader class): validation data
    loss_fn(class): loss function
    optimizer(class): optimization function
    save_name(string): name of the model, default None
    '''
    # starting time at the beginning of training
    time_start = time.time()


    # Steo 0: creating lists where the values of training and validation loss as well as the accuracy of training and validation are saved
    # the lists contain an epoch amount of 0's, and as the model calculates the values they fill up the list (history of learning)
    
    # training loss
    loss_train = [0]*n_epochs
    
    # training accuracy
    accuracy_train = [0]*n_epochs
    
    # validation loss
    loss_valid = [0]*n_epochs
    
    # validation accuracy
    accuracy_valid = [0]*n_epochs
    
    # location of saved model
    path = "/kaggle/working/model_"
    
    for epoch in range(n_epochs):
        
        #---------------
        #   TRAINING 
        #---------------
        
        # enabling training mode for the architecture
        model.train()
        
        # batch learning
        for n, batch in enumerate(train_loader):
            
            imgs, labels = batch

            # enable use of GPU or CPU + reshaping the labels list so it contains 1 column
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)

            # model predictions
            outputs = model(imgs)

            # calculating training loss
            loss = loss_fn(outputs, labels)

            # calculate gradient
            loss.backward()

            # optimize parameters
            optimizer.step()

            # set gradient to 0 to avoid accumulation
            optimizer.zero_grad()

            # saving loss values
            loss_train[epoch] += loss.item()*labels.size(0) # multiplying with labels.size(0) so we have the loss for the whole batch
            
            # calculating accuracy
            acc = accuracy(outputs, labels, 0.5)
            # saving accuracy
            accuracy_train[epoch] += acc

        #---------------
        #   VALIDATION 
        #---------------
        
        # enable validation mode for model
        model.eval()
        
        # no gradient: validation requires no gradient
        with torch.no_grad():
            
            for n, batches in enumerate(valid_loader):
                imgsV, labelsV = batches
                # enabling use of GPU or CPU, reshaping labels list
                imgsV, labelsV = imgsV.to(device), labelsV.unsqueeze(1).to(device)
                
                # model predictions
                outputsV = model(imgsV)
                
                # calculate loss
                lossV = loss_fn(outputsV, labelsV)
                   
                # save loss
                loss_valid[epoch] += lossV.item()*labelsV.size(0)
                
                # calculate accuracy
                acc = accuracy(outputsV, labelsV, 0.5)
                
                # save accuracy
                accuracy_valid[epoch] += acc        
        
        # average loss and accuracy values
        loss_train[epoch] /= len(train_loader.dataset)
        
        accuracy_train[epoch] /= len(train_loader.dataset)
        
        loss_valid[epoch] /= len(valid_loader.dataset)
        
        accuracy_valid[epoch] /= len(valid_loader.dataset)
        
        print(f'Epoch: {epoch+1}, training loss={loss_train[epoch]}, training accuracy={accuracy_train[epoch]}, valid loss={loss_valid[epoch]}, valid accuracy={accuracy_valid[epoch]}')
        
    # -----------------
    # SAVING THE MODEL
    # -----------------
    
    # if specified, save model
    if save_name:
        # location
        torch.save(model.state_dict(), path+save_name)
        
        print(path+save_name, "is saved!")
    
    # end of training and validation time
    time_end = time.time()
    
    print("Training time=", (time_end-time_start)/60, " minutes")
    
    # saving loss and accuracy values
    return loss_train, loss_valid, accuracy_train, accuracy_valid

# function 3: testing function

def testing(dataset, model):
    '''
    This function tests the model's performance.
    dataset(DataLoader): test dataset
    model(Class): trained model
    '''
    accu = 0
    
    out = torch.tensor([])
    y_true = torch.tensor([])
    
    model.eval()
    with torch.no_grad():
        for n, batches in enumerate(dataset):
            images, labels = batches
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            outputs = model(images)
            
            acc = accuracy(outputs, labels, 0.5)
            accu += acc
            
            y_true = torch.cat((y_true, labels.to('cpu')), 0)
            out = torch.cat((out, outputs.to('cpu')), 0)
    
    accuu = accu/len(dataset.dataset)
    print("Accuracy of test set is:", accuu)
            
    return out, y_true

# function 4: plotting function
def plot_training_results(model_hist, model_hist_exp=None):
    '''
    This function is used for plotting graphs which show the history of the model's learning and validation process
    
    model_hist(list):list which contains the history of values of the training and validation loss as well as the accuracy of the reference model
    model_hist_exp(list): lwhich contains the history of values of the training and validation loss as well as the accuracy 
    of the experimental model which we compare with the reference model, default is None
    '''
    
    plt.style.use('ggplot')
    plt.figure(figsize=(5,5))
    
    #------------------
    # REFERENCE MODEL
    #------------------
    
    #training loss
    plt.plot(model_hist[0], color="salmon", linestyle="dashdot")
    
    # validation loss
    plt.plot(model_hist[1], color="red")
    
    # training accuracy
    plt.plot(model_hist[2], color="deepskyblue", linestyle="dashdot")
    
    # validation accuracy
    plt.plot(model_hist[3], color="navy")
    
    # if experimental model specified
    if model_hist_exp:
        
        #--------------------
        # EXPERIMENTAL MODEL
        #--------------------

        #training loss exp
        plt.plot(model_hist_exp[0], color="saddlebrown", linestyle="dashed")

        # validation loss exp
        plt.plot(model_hist_exp[1], color="peru")

        # training accuracy exp
        plt.plot(model_hist_exp[2], color="yellowgreen", linestyle="dashed")

        # validation accuracy exp
        plt.plot(model_hist_exp[3], color="darkolivegreen")

    # x-axis label
    plt.xlabel("Epoch")
    # y-axis label
    plt.ylabel("Loss/Accuracy")
    # title
    plt.title("Loss/Accuracy history")
    # legend
    plt.legend()
    
    plt.show()

    