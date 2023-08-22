# Define dataset
import torch
import sys
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from torchvision.transforms import ToTensor, Compose, CenterCrop


class Collector: 
    def __init__(self):
        self.training_data = []
        self.testing_data = []
        self.eval_data = []

    def collect_data(self): 
        root_dir = '/home/hbi/RAD_NL25_RAP_5min/' 

        df_train = pd.read_csv('/users/hbi/taming-transformers/training_Delfland08-14_20.csv', header=None)
        event_times = df_train[0].to_list()
        dataset_train = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_s = pd.read_csv('/users/hbi/taming-transformers/training_Delfland08-14.csv', header=None)
        event_times = df_train_s[0].to_list()
        dataset_train_del = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_test = pd.read_csv('/users/hbi/taming-transformers/testing_Delfland18-20.csv', header=None)
        event_times = df_test[0].to_list()
        dataset_test = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_vali = pd.read_csv('/users/hbi/taming-transformers/validation_Delfland15-17.csv', header=None)
        event_times = df_vali[0].to_list()
        dataset_vali = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_aa = pd.read_csv('/users/hbi/taming-transformers/training_Aa08-14.csv', header=None)
        event_times = df_train_aa[0].to_list()
        dataset_train_aa = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_dw = pd.read_csv('/users/hbi/taming-transformers/training_Dwar08-14.csv', header=None)
        event_times = df_train_dw[0].to_list()
        dataset_train_dw = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_re = pd.read_csv('/users/hbi/taming-transformers/training_Regge08-14.csv', header=None)
        event_times = df_train_re[0].to_list()
        dataset_train_re = radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        data_list = [dataset_train_aa, dataset_train_dw, dataset_train_del, dataset_train_re]
        train_aadedwre = torch.utils.data.ConcatDataset(data_list)

        print(len(dataset_train), len(dataset_test), len(dataset_vali))
        loaders = {'train': DataLoader(train_aadedwre, batch_size=1, shuffle=True, num_workers=8),
                'test': DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8),
                'valid': DataLoader(dataset_vali, batch_size=1, shuffle=False, num_workers=8),

                'train_aa5': DataLoader(dataset_train_aa, batch_size=1, shuffle=False, num_workers=8),
                'train_dw5': DataLoader(dataset_train_dw, batch_size=1, shuffle=False, num_workers=8),
                'train_del5': DataLoader(dataset_train_del, batch_size=1, shuffle=True, num_workers=8),
                'train_re5': DataLoader(dataset_train_re, batch_size=1, shuffle=False, num_workers=8),
                }
        return loaders
    
    def collect_training_data(self):
        loaders=self.collect_data()

        for i, images in enumerate(loaders['train']):
            image = images.unsqueeze(2)
            #print(images.size())
            #print(image.size())
            self.training_data.append(image[0])

        return self.training_data

class radarDataset(Dataset):
    def __init__(self, root_dir, event_times, obs_number = 3, pred_number = 6, transform=None):
        # event_times is an array of starting time t(string)
        # transform is the preprocessing functions
        self.root_dir = root_dir
        self.transform = transform
        self.event_times = event_times
        self.obs_number = obs_number
        self.pred_number = pred_number
    def __len__(self):
        return len(self.event_times)
    def __getitem__(self, idx):
        start_time = str(self.event_times[idx])
        time_list_pre, time_list_obs = self.eventGeneration(start_time, self.obs_number, self.pred_number)
        output = []
        time_list = time_list_obs + time_list_pre
        #print(time_list)
        for time in time_list:
            year = time[0:4]
            month = time[4:6]
            #path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAC_MFBS_EM_5min_' + time + '_NL.h5'
            path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAP_5min_' + time + '.h5'
            image = np.array(h5py.File(path)['image1']['image_data'])
            #image = np.ma.masked_where(image == 65535, image)
            image = image[264:520,242:498]
            image[image == 65535] = 0
            image = image.astype('float32')
            image = image/100*12
            image = np.clip(image, 0, 128)
            image = image/40
            #image = 2*image-1 #normalize to [-1,1]
            output.append(image)
        output = torch.permute(torch.tensor(np.array(output)), (1, 2, 0))
        output = self.transform(np.array(output))
        return output

    def eventGeneration(self, start_time, obs_time = 3 ,lead_time = 6, time_interval = 30):
        # Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+1,...,t+72]]
        # Get the start year, month, day, hour, minute
        year = int(start_time[0:4])
        month = int(start_time[4:6])
        day = int(start_time[6:8])
        hour = int(start_time[8:10])
        minute = int(start_time[10:12])
        #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
        times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=time_interval * (x+1))) for x in range(lead_time)]
        lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
        times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=time_interval * x)) for x in range(obs_time)]
        obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
        obs.reverse()
        return lead, obs
