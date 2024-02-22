import os
import logging
import numpy as np

def get_mae(test, pred):
    result = []
    for i in range(len(test)):
        result.append(np.abs(test[i]-pred[i])[0])
    return sum(result)/len(result)

def get_rmse(test, pred):
    result = []
    for i in range(len(test)):
        result.append(np.square(test[i]-pred[i])[0])
    return np.sqrt(sum(result)/len(result))