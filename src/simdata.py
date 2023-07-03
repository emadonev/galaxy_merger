# importing everything
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import simple_norm
from sklearn.model_selection import train_test_split


# loading the data
hdu_pristine = fits.open('../input/pristine.fits') # opening the pristine dataset
hdu_noisy = fits.open('../input/noisy.fits')

# splitting the data into the images and labels part
x_pr = hdu_pristine[0].data
x_no = hdu_noisy[0].data

y_pr = hdu_pristine[1].data
y_no = hdu_noisy[1].data

# converting the float type to 32
x_pr = np.asarray(x_pr).astype('float32')
y_pr = np.asarray(y_pr).astype('float32')

x_no = np.asarray(x_no).astype('float32')
y_no = np.asarray(y_no).astype('float32')

# dividing the data into training, validation and testing sets
x_train_pr, x_rem_pr, y_train_pr, y_rem_pr = train_test_split(x_pr,y_pr, train_size=0.7, random_state=42, shuffle=True) # cutting off training data and the rest goes into the next stage
x_valid_pr, x_test_pr, y_valid_pr, y_test_pr = train_test_split(x_rem_pr,y_rem_pr, test_size=0.34, random_state=42, shuffle=True) # dividing the validation and testing sets

x_train_no, x_rem_no, y_train_no, y_rem_no = train_test_split(x_no,y_no, train_size=0.7, random_state=42, shuffle=True) # cutting off training data and the rest goes into the next stage
x_valid_no, x_test_no, y_valid_no, y_test_no = train_test_split(x_rem_no,y_rem_no, test_size=0.34, random_state=42, shuffle=True) # dividing the validation and testing sets

# converting the data format into tensors
x_train_pr, y_train_pr, x_valid_pr, y_valid_pr, x_test_pr, y_test_pr = map(torch.tensor, (x_train_pr, y_train_pr, x_valid_pr, y_valid_pr, x_test_pr, y_test_pr))
x_train_no, y_train_no, x_valid_no, y_valid_no, x_test_no, y_test_no = map(torch.tensor, (x_train_no, y_train_no, x_valid_no, y_valid_no, x_test_no, y_test_no))

# loading the batches
# pristine
def pristine_data(bs):
    train_ds_pr = TensorDataset(x_train_pr, y_train_pr)
    train_dl_pr = DataLoader(train_ds_pr, batch_size=bs, shuffle=True)

    valid_ds_pr = TensorDataset(x_valid_pr, y_valid_pr)
    valid_dl_pr = DataLoader(valid_ds_pr, batch_size=bs)

    test_ds_pr = TensorDataset(x_test_pr, y_test_pr)
    test_dl_pr = DataLoader(test_ds_pr, batch_size=bs)

    return train_dl_pr, valid_dl_pr, test_dl_pr


# noisy
def noisy_data(bs):
    train_ds_no = TensorDataset(x_train_no, y_train_no)
    train_dl_no = DataLoader(train_ds_no, batch_size=bs, shuffle=True)

    valid_ds_no = TensorDataset(x_valid_no, y_valid_no)
    valid_dl_no = DataLoader(valid_ds_no, batch_size=bs)

    test_ds_no = TensorDataset(x_test_no, y_test_no)
    test_dl_no = DataLoader(test_ds_no, batch_size=bs)

    return train_dl_no, valid_dl_no, test_dl_no