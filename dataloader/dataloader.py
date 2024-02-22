#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

class LiverDataset(Dataset):
    def __init__(self, data, y, min_max_img, min_max_r):
        self.data = data
        self.y = y
        self.min_max_img = min_max_img
        self.min_max_r = min_max_r
        
    def __getitem__(self, index):
        # image
        img = np.load(self.data[index])[:1].astype(np.float32)
        img[0] = (img[0] - self.min_max_img[0, 0])/(self.min_max_img[1, 0] - self.min_max_img[0, 0])
        
        # adj
        adj_path = self.data[index].replace('GridSat_B1_new_npy','GridSat_B1_new_kmean20/adj')
        adj_path = adj_path.replace('.npy', '.adj.pkl')
        with open(adj_path, "rb") as f:
            item = pickle.load(f)
        adj = item['adj'].astype(np.float32)
        
        # labels
        cluster_path = self.data[index].replace('GridSat_B1_new_npy','GridSat_B1_new_kmean20/cluster')
        cluster_path = cluster_path.replace('.npy', '.kmean20.pkl')
        with open(cluster_path, "rb") as f:
            item = pickle.load(f)
        labels = item['labels'].astype(np.int64)

        # reanalysis
        r_path = self.data[index].replace('GridSat_B1_new_npy', 'reanalysis_typhoon_npy')
        r_data = np.load(r_path).astype(np.float32)
        r_data[0] = (r_data[0] - self.min_max_r[0, 0])/(self.min_max_r[0, 1] - self.min_max_r[0, 0])
        r_data[1] = (r_data[1] - self.min_max_r[1, 0])/(self.min_max_r[1, 1] - self.min_max_r[1, 0])
        r_data[2] = (r_data[2] - self.min_max_r[2, 0])/(self.min_max_r[2, 1] - self.min_max_r[2, 0])
        
        
        return img, r_data, self.y[index].astype(np.float32), adj, labels

    def __len__(self):
        return len(self.data)


# In[3]:


class MyDataLoader:
    def __init__(self, config,):
        self.batch_size = int(config['data']['batch_size'])
        self.test_batch = int(config['data']['test_batch'])
        self.train_range = tuple(map(int, config['data']['train_range'].split(',')))
        self.test_range = tuple(map(int, config['data']['test_range'].split(',')))
        self.validate_ratio = float(config['data']['validate_ratio'])
        self.validate_random_state = int(config['data']['validate_random_state'])
        self.data_info = pd.read_csv(config['data']['windspeed_path'])
        self.min_max_img = np.load(config['data']['gridsat_min_max_path'])
        self.min_max_r = np.load(config['data']['reanalysis_min_max_path'])
    
    def train_loader(self,):
        
        self.train_info = self.data_info[(self.data_info["YEAR"]>=self.train_range[0]) & (self.data_info["YEAR"]<=self.train_range[1])]
        X = self.train_info["PATH"].to_numpy()
        Y = self.train_info["WIND_SPEED"].to_numpy()
        Y = np.expand_dims(Y, axis=1)
        train_valid_index = [*range(0, len(X))]

        train_index, self.valid_index = train_test_split(train_valid_index,test_size=self.validate_ratio, random_state=self.validate_random_state, shuffle=True)
        
        #form dataset
        train_indexset = LiverDataset(X[train_index], Y[train_index], self.min_max_img, self.min_max_r)

        #form dataloader
        train_loader = DataLoader(dataset=train_indexset,batch_size=self.batch_size,shuffle=True,)
        
        return train_loader
    
    def valid_loader(self, ):
        self.train_info = self.data_info[(self.data_info["YEAR"]>=self.train_range[0]) & (self.data_info["YEAR"]<=self.train_range[1])]
        X = self.train_info["PATH"].to_numpy()
        Y = self.train_info["WIND_SPEED"].to_numpy()
        Y = np.expand_dims(Y, axis=1)
        
        self.valid_indexset = LiverDataset(X[self.valid_index], Y[self.valid_index], self.min_max_img, self.min_max_r)
        valid_loader = DataLoader(dataset=self.valid_indexset,batch_size=self.batch_size,shuffle=True,)
        return valid_loader
    
    def test_loader(self, ):
        
        self.test_info = self.data_info[(self.data_info["YEAR"]>=self.test_range[0]) & (self.data_info["YEAR"]<=self.test_range[1])]

        X = self.test_info["PATH"].to_numpy()
        Y = self.test_info["WIND_SPEED"].to_numpy()
        Y = np.expand_dims(Y, axis=1)
        
        test_indexset = LiverDataset(X, Y, self.min_max_img, self.min_max_r)
        test_loader = DataLoader(dataset=test_indexset,batch_size=self.test_batch,shuffle=False,)
        return test_loader
