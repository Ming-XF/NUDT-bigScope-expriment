
import os
import pickle
import torch


class Config():
    exp_name = "5.pretrain"
    savedir = "../expriment/"+exp_name+"/"
    
    
    dataset_name = "pretrain"
    # Data flags
    #===================================================
    datadir = f"../data/{dataset_name}"
    trainset_name = f"pretrain_train.pkl"
    validset_name = f"pretrain_valid.pkl"
    testset_name = f"pretrain_test.pkl"
    
    retrain = True
    tb_log = False
    device = torch.device("cuda:0")
    
    max_epochs = 200
    batch_size = 128
    n_samples = 16
    
    init_seqlen = 10
    max_seqlen = 120
    min_seqlen = 20
    
    if dataset_name == "pretrain": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 3900
        lon_size = 2200
        sog_size = 1500
        cog_size = 72

        
        n_lat_embd = 512
        n_lon_embd = 512
        n_sog_embd = 512
        n_cog_embd = 512
    
        lat_min = 3
        lat_max = 42
        lon_min = 107
        lon_max = 129
        
        sog_max = 1500

    if dataset_name == "ct_dma": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13
    
    if dataset_name == "jun/ship": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        
        lat_min = 3
        lat_max = 42
        lon_min = 107
        lon_max = 129
        
        sog_max = 30
    
    if dataset_name == "jun/fly": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        
        lat_min = 3
        lat_max = 42
        lon_min = 107
        lon_max = 129
        
        sog_max = 1500
    
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