import sys 
sys.path.append("..") 

import os
import numpy as np
import pandas as pd
import math

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

import pdb

#第一种方式，分开近处和远处的矩阵
# def tra2mat(tra, mat_len):
#     mat = np.zeros((mat_len+1, mat_len+1))
#     tra = tra.reshape((-1, 2))
#     if len(tra) == 1:
#         mat[int(mat_len/2), int(mat_len/2)] = 1
#     else:
#         tra_max = np.max(tra, axis=0)
#         tra_min = np.min(tra, axis=0)
#         tra_range = tra_max - tra_min
#         index = ((tra - tra_min) / (tra_range + 1e-10) * (mat_len - 1)).astype(np.int32)
#         mat[index[:, 0], index[:, 1]] = 1
#     return mat

#第二种方式，近处和远处的矩阵都包含全局信息
# def tra2mat(all_tra, tra, mat_len):
#     mat = np.zeros((mat_len+1, mat_len+1))
#     tra_max = np.max(tra, axis=0)
#     tra_min = np.min(tra, axis=0)
#     tra_range = tra_max - tra_min
#     index = ((all_tra - tra_min) / (tra_range + 1e-10) * (mat_len - 1)).astype(np.int32)
#     index[index >= mat_len] = mat_len -1
#     index[index < 0] = 0
#     mat[index[:, 0], index[:, 1]] = 1

#     return mat


def getGaussianMap(center, grid_y, grid_x, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

#第三种方式，划分四部分空间矩阵, 4 x 500 x 500
# def getMat(geo):
#     if(len(geo) == 0):
#         return np.zeros((500, 500))
#     if(len(geo) == 1):
#         return getGaussianMap([250, 250], 500, 3)
#     geo_max = np.max(geo, axis=0)
#     geo_min = np.min(geo, axis=0)
#     geo_range = geo_max - geo_min
#     index = ((geo - geo_min) / (geo_range + 1e-10) * (500 - 1)).astype(np.int32)
#     heatmaps = []
#     for point in index:
#         heatmap = getGaussianMap(point, 500, 500, 3)
#         heatmaps.append(heatmap)
# #     mat[index[:, 0], index[:, 1]] = 1

#     pdb.set_trace()
#     return np.array(heatmaps)
    
    
# def getMats(geo):
    
#     geo_max = np.max(geo, axis=0)
#     geo_min = np.min(geo, axis=0)
#     mid = (geo_max - geo_min) / 2 + geo_min
#     mat1_points = geo[np.logical_and(geo[:, 0] < mid[0], geo[:, 1] < mid[1]), :]
#     mat2_points = geo[np.logical_and(geo[:, 0] < mid[0], geo[:, 1] > mid[1]), :]
#     mat3_points = geo[np.logical_and(geo[:, 0] > mid[0], geo[:, 1] < mid[1]), :]
#     mat4_points = geo[np.logical_and(geo[:, 0] > mid[0], geo[:, 1] > mid[1]), :]
#     mat1 = getMat(mat1_points)
#     mat2 = getMat(mat2_points)
#     mat3 = getMat(mat3_points)
#     mat4 = getMat(mat4_points)
#     mat = np.array([mat1, mat2, mat3, mat4])
    
#     return mat


#第四种方式，4 x 19 x 500 x 500
def getMat(geo, all_geo):
    if len(geo) == 0:
        return np.zeros((len(all_geo), 200, 200))
    
    geo_max = np.max(geo, axis=0)
    geo_min = np.min(geo, axis=0)
    geo_range = geo_max - geo_min
    
    heatmaps = []
    for point in all_geo:
        if point in geo:
            index = ((point - geo_min) / (geo_range + 1e-10) * (500 - 1)).astype(np.int32)
            heatmaps.append(getGaussianMap(index, 200, 200, 3))
        else:
            heatmaps.append(np.zeros((200, 200)))
    
    return np.array(heatmaps)
    
    
def getMats(geo):
    geo_max = np.max(geo, axis=0)
    geo_min = np.min(geo, axis=0)
    mid = (geo_max - geo_min) / 2 + geo_min
    mat1_points = geo[np.logical_and(geo[:, 0] < mid[0], geo[:, 1] < mid[1]), :]
    mat2_points = geo[np.logical_and(geo[:, 0] < mid[0], geo[:, 1] > mid[1]), :]
    mat3_points = geo[np.logical_and(geo[:, 0] > mid[0], geo[:, 1] < mid[1]), :]
    mat4_points = geo[np.logical_and(geo[:, 0] > mid[0], geo[:, 1] > mid[1]), :]
    mat1 = getMat(mat1_points, geo)
    mat2 = getMat(mat2_points, geo)
    mat3 = getMat(mat3_points, geo)
    mat4 = getMat(mat4_points, geo)
    mat = np.array([mat1, mat2, mat3, mat4])
    
    return mat
        


class Dataset_gowalla(Dataset):
    def __init__(self, data_path='../../data/gowalla/checkins-gowalla.txt', flag='train', seq_length=20, min_length=100,
                 timeenc=1, freq='h'):
        self.flag = flag
        self.seq_length = seq_length
        self.min_length = min_length
        
        self.timeenc = timeenc
        self.freq = freq
        
        self.lambda_t = 0.1
        self.lambda_s = 1000
        self.mat_len = 500
        
        self.scaler = None
        self.poi_len = None
        self.user_len = None
        
        self.data_path = data_path
        self.data = self.__read_data__()
        
        
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        seq_x = self.data[index][:-1, 3]
        seq_x_time = self.data[index][:-1, 4:8]
        seq_x_geo = self.data[index][:-1, 1:3]
        seq_x_slot = self.data[index][:-1, -1]
        
        seq_y = self.data[index][1:, 3]
        user_id = int(self.data[index][0][0])
        
         #计算空间矩阵
        #空间矩阵划分为四部分，并分别对每一部分进行放缩
        mat = getMats(seq_x_geo)
        
        #计算注意力FLash权重
        weight = torch.zeros((len(seq_x), len(seq_x)))
        seq_x_geo = torch.tensor(seq_x_geo)
        seq_x_slot = torch.tensor(seq_x_slot)
        for i in range(len(seq_x)):
            sum_w = 0
            for j in range(i+1):
                dist_t = seq_x_slot[i] - seq_x_slot[j]
                dist_s = torch.norm(seq_x_geo[i] - seq_x_geo[j], dim=-1)
                
                weight_t = ((torch.cos(dist_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(-(dist_t / 86400 * self.lambda_t))
                weight_s = torch.exp(-(dist_s * self.lambda_s))
                weight[i, j] = weight_s * weight_t + 1e-10 
                sum_w += weight[i, j]
            
            weight[i, :] = weight[i, :] / sum_w    
        
        
        
        return seq_x, seq_x_time, seq_y, user_id, weight, mat
        

    def __read_data__(self):
        #读取数据
        df = pd.read_table(self.data_path, sep='\t', names=['user', 'date', 'lat', 'lon', 'poi'])
        
        #提取出checkins数量大于等于self.min_length的用户
        data = []
        for index, item in df.groupby('user'):
            if len(item) >= self.min_length:
                data.append(item.to_numpy())
        
        #重新排列POI的ID
        self.user_len = len(data)
        data = np.concatenate(data, 0)
        poi = data[:, -1]
        
        poi_set = set(poi)
        self.poi_len = len(poi_set)
        poi_id = {}
        for (index, item) in enumerate(poi_set):
            poi_id[item] = index

        for i in range(len(data)):
            data[i][-1] = poi_id[data[i][-1]]
        
        #按照用户分组构造数据
        df = pd.DataFrame(data, columns=['user','date','lat','lon','poi'])
        df['date'] = pd.to_datetime(df['date']) #转换时间.apply(lambda x: x.value)
        data = []
        for (index, (userid, item)) in enumerate(df.groupby('user')):
            temp_data = item.to_numpy()
            temp_data[:, 0] = index #重新排列user ID
            data.append(temp_data)
        data = np.array(data)
        
        #划分训练和测试集
        for i in range(len(data)):
            if self.flag == 'train':
                data[i] = data[i][:int(0.8 * len(data[i])), :] #以8:2比例划分训练集
                data[i] = data[i][:len(data[i]) - len(data[i]) % self.seq_length, :] #删除不够组成self.seq_length条的多余数据
                data[i] = data[i].reshape(int(len(data[i])/self.seq_length), self.seq_length, 5) #重构数据，(user_num, trajectory_num, self.seq_length, 5)
            else:
                data[i] = data[i][int(0.8 * len(data[i]))+1:, :]
                data[i] = data[i][:len(data[i]) - len(data[i]) % self.seq_length, :]
                data[i] = data[i].reshape(int(len(data[i])/self.seq_length), self.seq_length, 5)
        
        #归一化
        data = np.concatenate(data, 0)
        data = np.concatenate(data, 0)
        date = data[:, 1].reshape(-1, 1)
        geo = data[:, [2, 3]].astype('float32')
        user_id = data[:, 0].reshape(-1, 1)
        poi_id = data[:, -1].reshape(-1, 1)
        
        #取消地理位置归一化
#         self.scaler = StandardScaler()
#         self.scaler.fit(geo)
#         geo = self.scaler.transform(geo)
        df_date = pd.DataFrame(date, columns=['date'])
        time_feature = time_features(df_date, timeenc=self.timeenc, freq=self.freq)
        
        #制作一周小时的时间插槽
        df_time_slot = df_date['date'].apply(lambda x: x.weekday()*24 + x.hour)
        time_slot = df_time_slot.values.reshape((-1, 1))
        
        #重构数据
        all_data = np.concatenate([user_id, geo, poi_id, time_feature, time_slot], 1)
        all_data = all_data.astype('float32')
        df = pd.DataFrame(all_data, columns=['user', 'lat', 'lon', 'poi', 'time1', 'time2', 'time3', 'time4', 'time_slot'])
        data = []
        for(userid, item) in df.groupby('user'):
            temp_data = item.to_numpy().reshape((-1, 20, 9))
            data.append(temp_data)
        data = np.concatenate(data, 0)
        
        return data
    
if __name__ == '__main__':
    dataset = Dataset_gowalla(flag="train")
    data_loader = DataLoader(dataset, batch_size=1)
    
#     count = 0
#     for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, weight, near_mat, far_mat) in enumerate(data_loader):
#         pdb.set_trace()
#         print('\tcurrent: {}\t | total: {}\n'.format(i, len(data_loader)))
#     pdb.set_trace()
#     print('\tresult: {}'.format(count / (len(data_loader) * 300 * 19)))
    
    
    #不划分近远矩阵，无重率为0.5多
    #划分近远矩阵，无重率为0.73
    
    
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, weight, mat) in enumerate(data_loader):
        pdb.set_trace()
    pdb.set_trace()
