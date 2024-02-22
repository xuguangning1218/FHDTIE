import os
import torch
import logging
import datetime
import builtins
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from model.FHDTIE import FHDTIE
from datetime import datetime
from utils.utils import get_mae, get_rmse

class Model():
    def __init__(self, config, dataloader, model_save_folder):
        self.device_str = str(config['data']['device'])
        self.num_gpus = int(config['data']['num_gpus'])            
        self.device = torch.device(self.device_str)
        if self.num_gpus > 1:
            self.model = nn.DataParallel(FHDTIE(config)).to(self.device)
        else:
            self.model = FHDTIE(config).to(self.device)
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.epochs = int(config['data']['epochs'])
        self.patient = int(config['data']['patient'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config['data']['learning_rate']))
        self.dataloader = dataloader
        self.model_save_folder = model_save_folder

    def setup_logger(self, model_save_folder):
        
        level =logging.INFO
    
        log_name = 'model.log'
    
        fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
        fileHandler.setLevel(logging.INFO)
    
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
    
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
    
        logger = logging.getLogger(model_save_folder + log_name)
        logger.setLevel(level)
    
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
    
        self.logger = logger

    def load(self, model_save_folder):
        model_save_path = '{}/best_validate_model.pth'.format(model_save_folder)
        checkpoint = torch.load(model_save_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['min_val_loss']
    
    def train(self, train_loader, valid_loader, start_epoch=0, min_val_loss = float('inf')):
        early_stop_counter = 0
        start = datetime.now()
        for epoch in range(start_epoch, self.epochs):
            self.logger.info('epochs [{}/{}]'.format(epoch + 1, self.epochs))
            starttime = datetime.now()
            total_train_loss = 0
            train_cnt = 0
            for _data, _r_data, _target, _adj, _labels in tqdm(train_loader):
                data = _data.to(self.device)
                r_data = _r_data.to(self.device)
                target = _target.to(self.device)
                labels = _labels.to(self.device)
                adj = _adj.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                pred, pre_labels = self.model(data, r_data, adj)
                loss1 = self.criterion(pred, target)
                loss2 = self.criterion2(pre_labels, labels)
                loss = loss1 + loss2
                total_train_loss += loss.item()
                train_cnt += 1
                loss.backward()
                self.optimizer.step()
                
            #############################################################################################
            # validate
            #############################################################################################
            with torch.no_grad():
                total_val_loss = 0
                val_cnt = 0
                for _data, _r_data, _target, _adj, _labels in tqdm(valid_loader):
                    data = _data.to(self.device)
                    r_data = _r_data.to(self.device)
                    target = _target.to(self.device)
                    labels = _labels.to(self.device)
                    adj = _adj.to(self.device)
                    pred, pre_labels = self.model(data, r_data, adj)
                    loss1 = self.criterion(pred, target)
                    loss2 = self.criterion2(pre_labels, labels)
                    loss = loss1 + loss2
                    val_cnt += 1
                    total_val_loss += loss.item()
        
                if min_val_loss > total_val_loss:
                    checkpoint = {
                         'epoch': epoch+1, # next epoch
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'min_val_loss': total_val_loss
                    }
                    torch.save(checkpoint,  self.model_save_folder + '/' + 'best_validate_model.pth')
                    self.logger.info('Saving Model in epoch {}'.format(epoch+1))
                    min_val_loss = total_val_loss       
                    early_stop_counter = 0
            endtime = datetime.now()
            self.logger.info('cost:{}s total_train_loss:{} val_loss: {}'.format((endtime - starttime).seconds,total_train_loss/train_cnt, total_val_loss/val_cnt))
        
            early_stop_counter += 1
            if early_stop_counter >= self.patient:
                logger.info('Early Stop Training')
                break
        end = datetime.now()
        self.logger.info('Finished Training')
        self.logger.info('Total cost:' + str((end-start).seconds)+ 's')
    
    def test(self, test_loader, model_save_path, builtins_print=False):

        if builtins_print == False:
            print = self.logger.info
        else:
            print = builtins.print

        paths = self.dataloader.test_info["PATH"].to_numpy()
        checkpoint = torch.load(model_save_path, map_location=self.device_str)
        self.model.load_state_dict(checkpoint['model'])
        
        test_list = []
        pred_list = []
        
        self.model.eval()
        with torch.no_grad():
            for _data, _r_data, _target, _adj, _labels in tqdm(test_loader):
                data = _data.to(self.device)
                r_data = _r_data.to(self.device)
                target = _target.to(self.device)
                labels = _labels.to(self.device)
                adj = _adj.to(self.device)
                pred, pre_labels = self.model(data, r_data, adj)
                target = _target.cpu().detach().numpy()
                pred_list.extend(pred.cpu().detach().numpy())
                test_list.extend(target)
        
        index_2018 = 0
        index_2019 = 0
        index_2020 = 0
        for i in range(len(paths)):
            if int(paths[i].split('/')[-2][:4])==2018:
                index_2018 = i
            elif int(paths[i].split('/')[-2][:4])==2019:
                index_2019 = i
            elif int(paths[i].split('/')[-2][:4])==2020:
                index_2020 = i
        index_2018 += 1
        index_2019 += 1
        index_2020 += 1
        
        mae_2018 = get_mae(test_list[:index_2018], pred_list[:index_2018])
        mae_2019 = get_mae(test_list[index_2018:index_2019], pred_list[index_2018:index_2019])
        mae_2020 = get_mae(test_list[index_2019:index_2020], pred_list[index_2019:index_2020])
        maes = get_mae(test_list, pred_list)
        print('MAE in 2018:'+str(mae_2018))
        print('MAE in 2019:'+str(mae_2019))
        print('MAE in 2020:'+str(mae_2020))
        print('MAE in avg:'+str(maes))
        
        rmse_2018 = get_rmse(test_list[:index_2018], pred_list[:index_2018])
        rmse_2019 = get_rmse(test_list[index_2018:index_2019], pred_list[index_2018:index_2019])
        rmse_2020 = get_rmse(test_list[index_2019:index_2020], pred_list[index_2019:index_2020])
        rmses = get_rmse(test_list, pred_list)
        print('RMSE in 2018:'+str(rmse_2018))
        print('RMSE in 2019:'+str(rmse_2019))
        print('RMSE in 2020:'+str(rmse_2020))
        print('RMSE in avg:'+str(rmses))