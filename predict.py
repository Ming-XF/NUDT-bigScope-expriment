
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
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
from config import Config

import warnings
warnings.filterwarnings('ignore')

cf = Config()

utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":
    
    expriment_name = "1.origin"
    
    data_path = "../data/ct_dma/ct_dma_test.pkl"
    model_path = "../expriment/" + expriment_name +"/model.pt"
    save_dir = "../expriment/" + expriment_name + "/prediction/"
    
    ## Data
    # ===============================
    moving_threshold = 0.05
    with open(data_path, "rb") as f:
        l_pred_errors = pickle.load(f)
    for V in l_pred_errors:
        try:
            moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
        except:
            moving_idx = len(V["traj"]) - 1
        V["traj"] = V["traj"][moving_idx:, :]
    data = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
    dataset = datasets.Predict_AISDataset(data, device=cf.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)
    model = model.to(cf.device)


    ## Load the best model
    # ===============================
    model.load_state_dict(torch.load(model_path))

    ## Save dir
    # ===============================
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ## Predict
    # ===============================
    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)

    model.eval()
    with torch.no_grad():
        for it, seqs in enumerate(dataloader):
            #历史轨迹点个数，最大不能超过min_seqlen
            history_len = 18
            #预测未来轨迹点个数
            predict_len = 50
            
            seqs_init = seqs[:, :history_len, :].to(cf.device)
            preds = trainers.sample(model,
                                    seqs_init,
                                    predict_len,
                                    temperature=1.0,
                                    sample=True,
                                    sample_mode=cf.sample_mode,
                                    r_vicinity=cf.r_vicinity,
                                    top_k=cf.top_k)
            inputs = seqs.to(cf.device)
            input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
            pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
            
            img_path = os.path.join(save_dir, f'{it}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()[0]
            inputs_np = seqs.detach().cpu().numpy()[0]

            
            plt.plot(inputs_np[history_len:, 1], inputs_np[history_len:, 0], linestyle="-.", color='red')
            plt.plot(preds_np[:history_len, 1], preds_np[:history_len, 0], "o", color='red')
            plt.plot(preds_np[history_len:, 1], preds_np[history_len:, 0], "x", markersize=4, color='blue')
            
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()
            
            pdb.set_trace()
            
            
            
            