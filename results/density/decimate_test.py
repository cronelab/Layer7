import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os,csv, time
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from pprint import pprint

import gc
import utils
from utils import *
import file_loader
import pn_demo_model as pn


import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------
# Constants 

probe_map_path =  './config/Precision_map_v2.xml'
DATA_ROOT = '/home/becca/Data/PN_preprocessed/' 

            #'D:/Becca/Data/PN_preprocessed/'


training_block_sets = { "NSR-004-002": ["Training block 1_green__loc4_7_1742409770184_250319_144250",
                                       "Training block 1_green__loc4_7_1742409770184_250319_144250",
                                       'Training block 2_green__loc8_1_1742410047212_250319_144727'],
                                         
                                         
                        "NSR-004-003": ['Training block 1_FIH0050blue__loc4_6_1744227575794_250409_153935', 
                                        'Training block 2_FIH0050blue__loc8_2_1744227839734_250409_154359',
                                        'Training block 3_FIH0050blue__loc4_2_1744228097783_250409_154817'] ,
                          
                          
                        "NSR-004-004": ['center-out-easy-block1_FIH0068__1_1753286802143_250723_160642', 
                                        'center-out-easy-block2_FIH0068__1_1753287064855_250723_161104', 
                                        'center-out-easy-block3_FIH0068__3_1753287429688_250723_161709',]  }


validate_block_sets = { "NSR-004-002": ["Validation block_green_loc8_2_1742410614837_250319_145654"],
                                         
                                         
                        "NSR-004-003": ['Validation block_FIH0050blue_loc8_3_1744228487511_250409_155447'] ,
                          
                          
                        "NSR-004-004": ['center-out-easy-block4_FIH0068__1_1753288012531_250723_162652']  }

hil_block_sets = { "NSR-004-002": ["Inference block_green_loc4_7_1742411124129_250319_150524"],
                                         
                                         
                        "NSR-004-003": ['Inference block_FIH0050blue_loc4_9_1744228833127_250409_160033'] ,
                          
                          
                        "NSR-004-004": ['center-out-easy-inference_FIH0068__2_1753288323373_250723_163203']  }



#config = pn.Config()

batch_size = 64
trial_length = 100
stride_length = 20


save_file = "P5_380um.csv"
numIter = 5
grid_size = 10
density = 1
#--------------------------------------------------
for grid_size in [30,25,20]:
    map = pos_to_electrode_map(probe_map_path)
    set_list = get_electrode_subsets(map, [grid_size], density) 
    #set_list = get_electrode_subsets_density(map, 9)

    with open(save_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)

        ## output headings 
        csvwriter.writerow(['Subject', 'R2', 'mse', 'gridsize', 'densityfactor', 'grid_pos' ])

    for subject in [102005]:
        config = Config(subject,use_general_params = False)
        config.update_num_epochs(60)
        #set_list = get_electrode_subsets(map, [grid_size], density) 
        #set_list = get_electrode_subsets_density(map, 9)

        #if(len(set_list) > 10):
        #    set_idxs = np.linspace(1,len(set_list)-1, 10).astype(int)
        #    print(set_idxs)
        #    set_list = [set_list[i] for i in set_idxs]
        set_count = 0

        # large electrodes only
        set_list = [ [
            [1,6,11,16,21,26,31],
            [3, 8, 13,18,23,28]
        ]]
        for set in set_list:
            N = len(set[0])
            M = len(set[1])
            config.update_grid_dims(N,M)
            grid_size_string = f'{N}x{M}'

            set_count+=1
            print(f'Set {set_count} / {len(set_list)}')

            print("Loading data...")
            train_loader, valid_loader, test_loader = get_dataloaders(config, x_set=set[0], y_set = set[1])
            print("...done")


            for iter in range(numIter):
                t0 = time.time()
                # make model
                print("building model")
                model = CNN2D_LSTM(config)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                criterion = torch.nn.MSELoss(reduction='sum')
                print("done")
                gc.collect()

                # train
                print("Training model")
                print(f'{config.epochs} epochs')
                CNN3D_train(model, optimizer, criterion, config, train_loader, valid_loader)
                gc.collect()
                print("done")

                # eval
                print("Testing...")
                tgts, preds = CNN3D_predict(model, test_loader)

                mse = np.linalg.norm(tgts-preds, axis=1)/2
                r2x = r2_score(tgts[:, 0], preds[:, 0])
                r2y = r2_score(tgts[:, 1], preds[:, 1])
                score = r2x + r2y
                error = np.linalg.norm(tgts - preds,axis=1)
                se = error**2
                mse = np.mean(se)
                gc.collect()
                print("done")
                
                print(f'Training completed in { (time.time()-t0) / 60} minutes')

                print("writing results...")
                
                with open(save_file, "a") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    # write to file
                    csvwriter.writerow([subject, score, mse, grid_size_string, density, set_count ])

                gc.collect()
