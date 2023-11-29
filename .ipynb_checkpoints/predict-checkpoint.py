
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils

# from cf_aire import Config
from cf_ship import Config

import warnings
warnings.filterwarnings('ignore')

cf = Config()

utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

def sample(traj, rate):
    output = []
    for index, item in enumerate(traj):
        if index == 0:
            output.append([item])
        else:
            last = output[-1][-1]
            last_time = last[-1]
            next_time = item[-1]
            
            if abs((next_time - last_time) - rate) < (rate / 3):
                output[-1].append(item)
            else:
                if ((next_time - last_time) - rate) < 0:
                    continue
                else:
                    output.append([item])
    return output


class Prediction():
    def __init__(self):
        self.model = models.TrAISformer(cf, partition_model=None)
        self.model = self.model.to(cf.device)
        # self.model.load_state_dict(torch.load(cf.ckpt_path))
        self.model.load_state_dict(torch.load(cf.nihe_path))
        self.model.eval()
        
        self.v_ranges = np.array([cf.lat_max-cf.lat_min, cf.lon_max-cf.lon_min, cf.sog_max, 360])
        self.v_roi_min = np.array([cf.lat_min, cf.lon_min, 0, 0])
        
    def preprocess(self, history):
        
        trajs = sample(history, cf.rate)
        output = []
        for item2 in trajs:
            if len(item2) >= (cf.min_seqlen / 2):
                item3 = np.array(item2)

                cog = item3[:, 3]
                cog[cog < 0] = 0
                cog[cog > 360] = 360
                item3[:, 3] = cog

                sog = item3[:, 2]
                sog[sog < 0] = 0
                item3[:, 2] = sog

                lat = item3[:, 0]
                lat[lat < -90] = -90
                lat[lat > 90] = 90
                item3[:, 0] = lat

                lon = item3[:, 1]
                lon[lon < -180] = -180
                lon[lon > 180] = 180
                item3[:, 1] = lon
                
                output.append(item3)
        
        if len(output) <= 0:
            print('error: 输入的轨迹经过采样后轨迹点个数不足')
            return []
        else:
            return output[0][-cf.max_seqlen:]


    def predict(self, history, predict_len=10):
        #接受参数：history，历史轨迹，是一个二维数组（len， 4）；predict_len，预测长度
        #返回参数：preds_np，预测轨迹，也是一个二维数组(len + predict_len, 4)
        
        history = self.preprocess(history)
        if len(history) != 0:
            history = history[:, :4]
            seqs = history.reshape((1, -1, 4))
            seqs = (seqs - self.v_roi_min) / self.v_ranges
            with torch.no_grad():
                seqs_init = torch.tensor(seqs).to(cf.device)
                preds = trainers.sample(self.model,
                                        seqs_init,
                                        predict_len,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                preds = preds.detach().cpu().numpy()[0]
                pred_coords = (preds * self.v_ranges + self.v_roi_min)
                return history, pred_coords
        else:
            return [], []


if __name__ == "__main__":
    
    SOG = 'speed'
    COG = 'track'
    LON = 'longitude'
    LAT = 'latitude'
    TIME = 'pos_datetime'
    FL = 'flight_line'
    
    # ID = 'aircraft_icao'
    # PATH = "../data/data_jw/aire"
    ID = 'mmsi'
    PATH = "../data/data_jw/vessel"
    
    prediction = Prediction()
    
    for filename in os.listdir(PATH):
        df = pd.read_csv(os.path.join(PATH, filename))
        df = df[[ID, FL, LON, LAT, SOG, COG, TIME]]

        for (index, (fl, item)) in enumerate(df.groupby(FL)):
            item = item[[LAT, LON, SOG, COG, TIME]]
            item = item.sort_values(by=TIME, ascending=True)
            temp = item.values

            seqs, pred = prediction.predict(temp, 10)
            
            if len(pred) > 0:
                pdb.set_trace()
                img_path = os.path.join('./test.jpg')
                plt.figure(figsize=(9, 6), dpi=150)
                plt.plot(seqs[:, 1], seqs[:, 0], linestyle="-.", color='red')
                plt.plot(pred[:, 1], pred[:, 0], linestyle=":", color='blue')
                plt.savefig(img_path, dpi=150)
                plt.close()
            
            
    
    
#     expriment_name = "6.pretrain-ship"
    
#     data_path = "../data/ct_dma/ct_dma_test.pkl"
#     model_path = "../expriment/" + expriment_name +"/model.pt"
#     save_dir = "../expriment/" + expriment_name + "/prediction/"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     ## Data
#     # ===============================
#     moving_threshold = 0.05
#     with open(data_path, "rb") as f:
#         l_pred_errors = pickle.load(f)
#     for V in l_pred_errors:
#         try:
#             moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
#         except:
#             moving_idx = len(V["traj"]) - 1
#         V["traj"] = V["traj"][moving_idx:, :]
#     data = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
#     dataset = datasets.Predict_AISDataset(data, device=cf.device)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    
    
    
#     prediction = Prediction()
#     for it, seqs in enumerate(dataloader):
#         pdb.set_trace()
#         data = [[118.47224693030826, 21.444319540598737,0.3,0.4], [118.1491119360207, 21.08887006075266,0.5,0.6], [118.1491119360207, 20.507228057164806,0.6,0.9], [118.1491119360207, 20.02424974140956,0.1,0.3], [117.92694055752229, 19.67512328948342,0.4,0.6], [117.58102092362748, 19.361311606934613,0.7,0.4], [116.52154158613932, 18.368189022325826,0.3,0.6], [119.9318374393007, 20.42586105943627,0.7,0.4], [119.7629258159244, 19.862436623682974,0.4,0.6], [118.95360260678594, 17.587000885125466,0.7,0.2]]
#         data = np.array(data)
#         seqs = data[[1, 0, 2, 3]]
# #         seqs = np.array(seqs)[0]
#         seqs = seqs * prediction.v_ranges + prediction.v_roi_min
# #         pred = prediction.predict(seqs[:18], predict_len=50)
#         pred = prediction.predict(seqs[:18], predict_len=50)
        
#         img_path = os.path.join(save_dir, f'{it}.jpg')
#         plt.figure(figsize=(9, 6), dpi=150)
#         cmap = plt.cm.get_cmap("jet")
        
#         plt.plot(seqs[18:, 1], seqs[18:, 0], linestyle="-.", color='red')
#         plt.plot(pred[:18, 1], pred[:18, 0], "o", color='red')
#         plt.plot(pred[18:, 1], pred[18:, 0], "x", markersize=4, color='blue')
        
#         plt.xlim([prediction.model.lon_min, prediction.model.lon_max])
#         plt.ylim([prediction.model.lat_min, prediction.model.lat_max])
#         plt.savefig(img_path, dpi=150)
#         plt.close()
        
#         pdb.set_trace()
        
    
            
            
            