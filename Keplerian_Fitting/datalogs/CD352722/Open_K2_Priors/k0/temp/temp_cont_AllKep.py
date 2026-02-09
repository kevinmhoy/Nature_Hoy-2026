import numpy as np
import pandas as pd
import pickle
import time
import os
nan = np.nan
gaussian_mixture_objects = dict()
A_ = []
mod_fixed_ = []
cornums = None

# BEGIN WRITE_DATA_RV FROM MODEL
my_data = pd.read_csv('datalogs/CD352722/run_5/k0/temp/temp_datacont_AllKep.csv', index_col=0)

X_ = my_data['BJD'].values
Y_ = my_data['RV'].values
YERR_ = my_data['eRV'].values
ndat = len(X_)



def my_model(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])

    model0 = np.zeros(ndat)
    err20 = YERR_ ** 2

    return model0, err20


