
import os
import pickle
import torch


class Config():
    exp_name = "jw_aire"
    savedir = "../exp/"+exp_name+"/"
    
    dataset_name = "ship"
    # Data flags
    #===================================================
    datadir = f"../data/"
    trainset_name = "{}_train.pkl".format(dataset_name)
    validset_name = "{}_valid.pkl".format(dataset_name)
    testset_name = "{}_test.pkl".format(dataset_name)

    retrain = True
    tb_log = False
    device = torch.device("cuda:0")
    
    max_epochs = 200
    batch_size = 128
    n_samples = 10
    
    init_seqlen = 10
    max_seqlen = 60
    min_seqlen = 20
    
    if dataset_name == "ship": #==============================
        
        rate = 60 * 30
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 10000
        lon_size = 10000
        sog_size = 50
        cog_size = 180

        
        n_lat_embd = 512
        n_lon_embd = 512
        n_sog_embd = 128
        n_cog_embd = 128
    
        lat_min = -27
        lat_max = 43
        lon_min = 73
        lon_max = 176
        
        sog_max = 50

    if dataset_name == "aire": #==============================
        
        rate = 60 * 2.5
        
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 10000
        lon_size = 10000
        sog_size = 4600
        cog_size = 180

        
        n_lat_embd = 512
        n_lon_embd = 512
        n_sog_embd = 128
        n_cog_embd = 128
    
        lat_min = -36
        lat_max = 65
        lon_min =  -180
        lon_max = 180
        
        sog_max = 4600
    
    #===========================================================================
    # Model and sampling flags
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1", 
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"
    top_k = 10 # int or None 
    r_vicinity = 40 # int
    
    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    
    # model parameters
    #===================================================
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 4 # for DataLoader
    
    ckpt_path = os.path.join(savedir,"model.pt")
    nihe_path = os.path.join(savedir,"model_200.pt")   