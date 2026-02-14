from data_provider .data_loader import Dataset_ETT_hour ,Dataset_ETT_minute ,Dataset_Custom ,Dataset_M4 
from data_provider .data_loader import Dataset_ETT_Exog 
from torch .utils .data import DataLoader 
import torch 
import random 
import numpy as np 

data_dict ={
'ETTh1':Dataset_ETT_hour ,
'ETTh2':Dataset_ETT_hour ,
'ETTm1':Dataset_ETT_minute ,
'ETTm2':Dataset_ETT_minute ,
'ECL':Dataset_Custom ,
'Traffic':Dataset_Custom ,
'Weather':Dataset_Custom ,
'm4':Dataset_M4 ,
'ETT_exog':Dataset_ETT_Exog ,
}


def _worker_init_fn (worker_id :int ):
    """Initialize seeds inside each DataLoader worker for reproducibility."""
    # Derive each worker's seed from the PyTorch initial seed to avoid collisions
    base_seed =torch .initial_seed ()%2 **32 
    np .random .seed (base_seed )
    random .seed (base_seed )


def data_provider (args ,flag ):
    Data =data_dict [args .data ]
    timeenc =0 if args .embed !='timeF'else 1 
    percent =args .percent 

    # debug output：confirmtimeencset up
    print (f"[DEBUG] args.embed = {args .embed }, timeenc = {timeenc }")

    if flag =='test':
        shuffle_flag =False 
        drop_last =True 
        batch_size =args .batch_size 
        freq =args .freq 
    else :
        shuffle_flag =True 
        drop_last =True 
        batch_size =args .batch_size 
        freq =args .freq 

    if args .data =='m4':
        drop_last =False 
        data_set =Data (
        root_path =args .root_path ,
        data_path =args .data_path ,
        flag =flag ,
        size =[args .seq_len ,args .label_len ,args .pred_len ],
        features =args .features ,
        target =args .target ,
        timeenc =timeenc ,
        freq =freq ,
        seasonal_patterns =args .seasonal_patterns 
        )
    elif args .data =='ETT_exog':
        data_set =Data (
        root_path =args .root_path ,
        data_path =args .data_path ,
        flag =flag ,
        size =[args .seq_len ,args .label_len ,args .pred_len ],
        features =args .features ,
        target =args .target ,
        scale =args .scale ,
        timeenc =timeenc ,
        freq =freq ,
        percent =percent 
        )
    else :
        data_set =Data (
        root_path =args .root_path ,
        data_path =args .data_path ,
        flag =flag ,
        size =[args .seq_len ,args .label_len ,args .pred_len ],
        features =args .features ,
        target =args .target ,
        scale =args .scale ,
        timeenc =timeenc ,
        freq =freq ,
        percent =percent ,
        seasonal_patterns =args .seasonal_patterns 
        )
        # Build a deterministic generator if provided; fall back to a fixed seed
    gen =torch .Generator ()
    try :
    # Reuse the global seed from args if available; default to 2021
        seed =getattr (args ,'seed',2021 )
    except Exception :
        seed =2021 
    gen .manual_seed (seed )

    data_loader =DataLoader (
    data_set ,
    batch_size =batch_size ,
    shuffle =shuffle_flag ,
    num_workers =args .num_workers ,
    drop_last =drop_last ,
    worker_init_fn =_worker_init_fn if args .num_workers and args .num_workers >0 else None ,
    generator =gen )
    return data_set ,data_loader 
