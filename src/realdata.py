# importing everything
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split

import h5py # for manipulating h5 files
from skimage.transform import resize # for resizing the images

# defining the filepath to the dataset
filename = '../input/Galaxy10_DECals.h5'
# loading the data
hf = h5py.File(filename, 'r+')
# assigning the data
images_r = hf['images'] # images
labels_r = hf['ans'] # labels

# changing the labels to be in 2 classes
for i, element in enumerate(labels_r):
    if element != 1:
        labels_r[i] = 0

# resizing the images
# creating an empy array to store all of the resized images
resized_images = np.empty((len(images_r), 75, 75, 3))

# resizing the images using scikit-image
for i, image in enumerate(images_r):
    resized_images[i] = resize(image, (75,75), mode='reflect')

# converting to float32
images_real = resized_images.astype(np.float32)
labels_real = labels_r.astype(np.float32)

# splitting the data
x_train, x_rem, y_train, y_rem = train_test_split(images_real,labels_real, train_size=0.7, random_state=42, shuffle=True) # cutting off training data and the rest goes into the next stage
x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.34, random_state=42, shuffle=True) # dividing the validation and testing sets

# conversion to Tensors
x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))

# loading the batches
def real_data(bs):
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)

    return train_dl, valid_dl, test_dl
