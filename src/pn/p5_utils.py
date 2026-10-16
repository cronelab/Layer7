
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


#----------------------------------------------------------
# Constants
#----------------------------------------------------------
SESSION_TIMES = np.asarray([22, 24+8.5, 24+12, 24+17, 24+19, 24+22, 48+10, 48+12, 48+18, 48+23, 72+9, 72+10])
SESSION_TIMES = SESSION_TIMES-SESSION_TIMES[0]

subject_root = "D:\\Becca\\Data\\PN\\NSR-102-005\\"

train_block_id = {
            165: "Session2\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_1_1767103993233_251230_141313\\",
            166: "Session2\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_2_1767104299586_251230_141819",
            167: "Session2\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_3_1767104515800_251230_142155",
            168: "Session2\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_4_1767104746342_251230_142546",
            173: "Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_1_1767117086679_251230_175126",
            174: "Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_2_1767117317127_251230_175517",
            175: "Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_3_1767117503028_251230_175823",
            176: "Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_4_1767117701812_251230_180141",
            177: "Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_5_1767117907782_251230_180507",
            188: "Session4\\29-DEC-2025\\webgrid-easy_FIH0117_train_1_1767129389894_251230_211629",
            189: "Session4\\29-DEC-2025\\webgrid-easy_FIH0117_train_2_1767129551453_251230_211911",
            190: "Session4\\29-DEC-2025\\webgrid-easy_FIH0117_train_3_1767129740140_251230_212220",
            201: "Session5\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767139067812_251230_235747",
            203: "Session5\\29-DEC-2025\\webgrid-medium_FIH0117_train_3_1767139362911_251231_000242",

            205: "Session5\\29-DEC-2025\\webgrid-easy_FIH0117_infer_2_1767138403593_251230_234643",

            211: "Session6\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767150443012_251231_030723",
            212: "Session6\\29-DEC-2025\\webgrid-medium_FIH0117_train_2_1767150632010_251231_031032",
            221: "Session7\\29-DEC-2025\\webgrid-hard_FIH0117_train_1_1767192254968_251231_144414",
            222: "Session7\\29-DEC-2025\\webgrid-hard_FIH0117_train_2_1767192551361_251231_144911",
            226: "Session7\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767194331798_251231_151851",
            227: "Session7\\29-DEC-2025\\webgrid-medium_FIH0117_train_2_1767194526858_251231_152206",
            231: "Session8\\29-DEC-2025\\webgrid-hard_FIH0117_train_1_1767203809930_251231_175649",
            232: "Session8\\29-DEC-2025\\webgrid-hard_FIH0117_train_2_1767204034506_251231_180034",
            233: "Session8\\29-DEC-2025\\webgrid-hard_FIH0117_train_3_1767204239804_251231_180359",
            234: "Session8\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767204862210_251231_181422",
            235: "Session8\\29-DEC-2025\\webgrid-medium_FIH0117_train_2_1767205076652_251231_181756",
            252: "Session9\\29-DEC-2025\\webgrid-hard_FIH0117_train_3_1767222994808_251231_231634",
            253: "Session9\\29-DEC-2025\\webgrid-hard_FIH0117_train_4_1767223204299_251231_232004",
            254: "Session9\\29-DEC-2025\\webgrid-hard_FIH0117_train_5_1767223412173_251231_232332",
            }


infer_blocks = {
                169: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session2\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767105022887_251230_143022\\",
                170: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session2\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767105453895_251230_143733\\",
                178: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session3\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767118172455_251230_180932",
                191: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session4\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767129891132_251230_212451\\",
                205: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session5\\29-DEC-2025\\webgrid-medium_FIH0117_infer_2_1767140835594_251231_002715\\", #206
                213: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767151073946_251231_031753\\",  # Session 6 has some no joystick inf
                223: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session7\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767193180404_251231_145940\\", #minimal joystick use
                236: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767206977752_251231_184937\\",
                266: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-medium_FIH0117_infer_2_1767227281175_260101_002801\\"
}

nojoy_blocks = {

                215: 'D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-easy_FIH0117_infer_2_1767151847540_251231_033047\\',
                214: 'D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767151400299_251231_032320\\',
                
                237: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_2_1767207322061_251231_185522", # + hand movement
                238: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_3_1767207662584_251231_190102", # + hand movement
                239: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_4_1767207953029_251231_190553\\",

                259: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767224678966_251231_234438\\",
                260: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767226373838_260101_001253\\",
                261: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_2_1767225390165_251231_235630\\",
                262: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_3_1767225709295_260101_000149\\",
                263: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_4_1767226045714_260101_000725\\",
               
}

stale_blocks = {
        198: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session5\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767138083419_251230_234123\\",  # 200
        210: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767150106991_251231_030147\\",
        220: "D:\\Becca\\Data\\PN\\NSR-102-005\\Session7\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767191902811_251231_143822\\",
}


infer_train_sets =  {
                    169: [165,166,167],                                                 # S2
                    170: [165,166,167],                                                 # S2
                    178: [165,166,167,168,173,174],                                     # S3
                    179: [174,175,176,177],                                             # S3
                    191: [175, 176, 177, 188],                                          # S4
                    192: [175, 176, 177, 188, 189, 190],                                # S4
                    198: [174,175,176,177],                                             # S5, stale
                    204: [188, 189, 190, 198, 201, 203],                                # S5
                    205: [175, 176, 177, 188, 189, 190, 203],                           # S6
                    213: [177, 188, 189, 190, 201, 203, 211, 212],                      # S6
                    223: [177, 190, 201, 203, 211, 221, 222],                           # S7
                    236: [221, 222, 226, 227, 231, 232, 233, 234, 235],                 # S8
                    258: [221, 222, 226, 227, 231, 232, 233, 234, 235, 252, 253, 254],   # S9
                    266: [221, 222, 226, 227, 231, 232, 233, 234, 235, 252, 253, 254],   # S9

}

#----------------------------------------------------------
# Enum
#----------------------------------------------------------
class  Outlist(Enum): 
    STATUS = int(0)
    TIME = int(1)
    SESSION = int(2)
    TTYPE =  int(3)
    LEVEL = int(4)
    STIME = int(5)


#----------------------------------------------------------
# Functions 
#----------------------------------------------------------
def get_block_time(blockroot):
    try: 
        labeldata = torch.load(os.path.join(blockroot, "CursorJoystickCRNNImplementation","labels.pt") )
        return labeldata.shape[0] / 100
    except:
        print(f'Error with {blockroot}')

def get_inf_train_time(blocknum):
    training_time = 0
    for block in infer_train_sets[blocknum]:
        block_path = train_block_id[block]
        training_time += get_block_time(os.path.join(subject_root, block_path))
        #print(f'used block {block}, net time {training_time}s, {training_time/60}min')
    return training_time

def pull_data_from_file(data_root, output_lists, trial_type, FAILURE_TIME_THRSH = 60):

    session = int(data_root[36])

    if("easy" in data_root) : level = "Easy"
    elif("medium" in data_root): level = "Medium"
    elif("hard" in data_root): level = "Hard"
    else:
        raise ValueError(f"The difficulty level is not valid.")

    if (not os.path.exists(os.path.join(data_root,'task_ui_events.jsonl'))):
        print( f'File {os.path.join(data_root,'task_ui_events.jsonl')} Does Not Exist')
        return
    with open( os.path.join(data_root,'task_ui_events.jsonl'), 'r') as json_file:
        json_list = list(json_file)

    t0 = None
    state = "REST"

    #try:
    for json_item in json_list:
        result = json.loads(json_item)
        if('type' in result.keys()):
            if (result['type'] == 'event'):
                if(result['data']['type'] == 'target_initialized'):
                    match state:
                        case "REST":
                            t0 = result['microsecond']
                            state = "TRIAL"
                        case "TRIAL":
                            output_lists[Outlist.STATUS.value].append("FAIL")
                            output_lists[Outlist.TIME.value].append(-1)
                            output_lists[Outlist.SESSION.value].append(session)
                            output_lists[Outlist.TTYPE.value].append(trial_type)
                            output_lists[Outlist.LEVEL.value].append(level)
                            output_lists[Outlist.STIME.value].append(SESSION_TIMES[session])

                            t0 = result['microsecond']

                        case _: 
                            raise ValueError(f"The state {state} is not valid.")

                elif(result['data']['type'] == 'targets_complete'):
                    match state:
                        case "REST":
                            raise ValueError(f"SHOULD NOT BE IN REST")
                        case "TRIAL":
                            dt = (result['microsecond'] - t0)/1000000
                            if(dt < FAILURE_TIME_THRSH):
                                #print(session, type(session))
                                output_lists[Outlist.STATUS.value].append("SUCCESS")
                                output_lists[Outlist.TIME.value].append(dt)
                                output_lists[Outlist.SESSION.value].append(session)
                                output_lists[Outlist.TTYPE.value].append(trial_type)
                                output_lists[Outlist.LEVEL.value].append(level)
                                output_lists[Outlist.STIME.value].append(SESSION_TIMES[session])

                            else:
                                output_lists[Outlist.STATUS.value].append("FAIL")
                                output_lists[Outlist.TIME.value].append(dt)
                                output_lists[Outlist.SESSION.value].append(session)
                                output_lists[Outlist.TTYPE.value].append(trial_type)
                                output_lists[Outlist.LEVEL.value].append(level)
                                output_lists[Outlist.STIME.value].append(SESSION_TIMES[session])

                            t0 = None
                            state = "REST"
                        case _: 
                            raise ValueError(f"The state {state} is not valid.")
    if(output_lists[Outlist.STATUS.value][-1] == "FAIL"):
        output_lists[Outlist.STATUS.value].pop()
        output_lists[Outlist.TIME.value].pop()
        output_lists[Outlist.SESSION.value].pop()
        output_lists[Outlist.TTYPE.value].pop()
        output_lists[Outlist.LEVEL.value].pop()
        output_lists[Outlist.STIME.value].pop()
    #except:
    #    print("Issue Hit")   


def calc_results(FAILURE_TIME_THRSH = 60):

        status_list = []
        time_list = []
        session_list = []
        type_list = []
        level_list = []
        stime_list = []

        train_list = [
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session2\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_1_1767103993233_251230_141313\\",  #165
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session3\\29-DEC-2025\\webgrid-medium-random_FIH0117_train_2_1767117317127_251230_175517\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session4\\29-DEC-2025\\webgrid-easy_FIH0117_train_1_1767129389894_251230_211629\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session5\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767139067812_251230_235747\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767150443012_251231_030723\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_train_1_1767150443012_251231_030723\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session7\\29-DEC-2025\\webgrid-hard_FIH0117_train_1_1767192254968_251231_144414\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-hard_FIH0117_train_1_1767203809930_251231_175649\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-hard_FIH0117_train_3_1767222994808_251231_231634\\",
                ]


        infer_list = [
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session2\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767105022887_251230_143022\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session2\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767105453895_251230_143733\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session3\\29-DEC-2025\\webgrid-easy-random_FIH0117_infer_1_1767118172455_251230_180932",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session4\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767129891132_251230_212451\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session5\\29-DEC-2025\\webgrid-medium_FIH0117_infer_2_1767140835594_251231_002715\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767151073946_251231_031753\\",  # Session 6 has some no joystick inf
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767151073946_251231_031753\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session7\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767193180404_251231_145940\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767206977752_251231_184937\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-medium_FIH0117_infer_2_1767227281175_260101_002801\\"
                ]


        nojoy_list = [
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-medium_FIH0117_infer_2_1767227281175_260101_002801\\",

                'D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-easy_FIH0117_infer_2_1767151847540_251231_033047\\',
                'D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767151400299_251231_032320\\',
                
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_2_1767207322061_251231_185522", # + hand movement
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_3_1767207662584_251231_190102", # + hand movement
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session8\\29-DEC-2025\\webgrid-easy_FIH0117_infer_4_1767207953029_251231_190553\\",

                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767224678966_251231_234438\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767226373838_260101_001253\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_2_1767225390165_251231_235630\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_3_1767225709295_260101_000149\\",
                "D:\\Becca\\Data\\PN\\NSR-102-005\\Session9\\29-DEC-2025\\webgrid-veasy_FIH0117_infer_4_1767226045714_260101_000725\\",
        ]

        oldmodel_list = [
        "D:\\Becca\\Data\\PN\\NSR-102-005\\Session5\\29-DEC-2025\\webgrid-easy_FIH0117_infer_1_1767138083419_251230_234123\\",
        "D:\\Becca\\Data\\PN\\NSR-102-005\\Session6\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767150106991_251231_030147\\",
        "D:\\Becca\\Data\\PN\\NSR-102-005\\Session7\\29-DEC-2025\\webgrid-medium_FIH0117_infer_1_1767191902811_251231_143822\\",
        ]


        for data_root in train_list:
                pull_data_from_file(    data_root, 
                                [status_list,
                                        time_list,
                                        session_list,
                                        type_list,
                                        level_list,
                                        stime_list ], 
                                        "TRAIN")
                
        for data_root in infer_list:
                pull_data_from_file(    data_root, 
                                [status_list,
                                        time_list,
                                        session_list,
                                        type_list,
                                        level_list,
                                        stime_list ], 
                                        "INFER")
        for data_root in nojoy_list:
                pull_data_from_file(data_root, 
                                [status_list,
                                        time_list,
                                        session_list,
                                        type_list,
                                        level_list,
                                        stime_list ], 
                                        "NO JOYSTICK")
                
        for data_root in oldmodel_list:
                pull_data_from_file(    data_root, 
                                [status_list,
                                        time_list,
                                        session_list,
                                        type_list,
                                        level_list,
                                        stime_list ], 
                                        "STALE MODEL")
        df = pd.DataFrame({
                        'Session': session_list,
                        'time': time_list,
                        'Block Type': type_list,
                        'status': status_list,
                        'Difficulty': level_list,
                        'Session Time': stime_list              
                        })
        df["Session"] = df["Session"].astype(int)
        return df


def split_preprocessed_pn(tgts, preds, thrsh = 0.15, plot_thrsh=True):
    dx = tgts[:,0]
    dy = tgts[:,1]

    mvmt = dx**2 + dy**2 
    T = len(mvmt)
    if(plot_thrsh):
        fig, ax = plt.subplots(1,1, figsize = (15,5));
        plt.plot(mvmt)
        plt.plot([0,T], [thrsh, thrsh])

    
    ## Split into trials 
    start_idxs = []
    end_idxs = []

    T = len(mvmt)
    mstate = "rest"
    for t in range(T):
        match mstate:
            case "rest":
                if mvmt[t] > thrsh:
                    start_idxs.append(t)
                    mstate = "moving"
            case "moving":
                if mvmt[t] < thrsh:
                    end_idxs.append(t)
                    mstate = "rest"
            case _:
                print("invalid state")
    x_values = []
    y_values = []
    angle_bins = []
    x_hat_values = []
    y_hat_values = []

    

    for trial in range(len(start_idxs)):
        t1 = start_idxs[trial]
        t2 = end_idxs[trial]
    
        dx = tgts[t1 : t2,0],#actove dx
        dx = np.squeeze(dx) 
    
        dy = tgts[t1:t2,1], #active dy
        dy = np.squeeze(dy)
        
        dx_hat = preds[t1 : t2,0]
        dx_hat = np.squeeze(dx_hat)
        dy_hat = preds[t1 : t2,1]
        dy_hat = np.squeeze(dy_hat)

        x_ =  np.cumsum(dx)  
        y_ =  np.cumsum(dy)  
        xhat_ =  np.cumsum(dx_hat)  
        yhat_ =  np.cumsum(dy_hat) 
    
        angle = 180/np.pi* np.atan2(  y_[-1] - y_[0], x_[-1] - x_[0] ) 
    
        angle_class = int( np.rint(angle / 45) + 4)
        if angle_class == 8:
            angle_class = 0

        x_values.append(x_)
        y_values.append(y_)    
        x_hat_values.append(xhat_)
        y_hat_values.append(yhat_)
        angle_bins.append(angle_class)

    return {'x_values': x_values,
            'y_values': y_values,
            'x_hat_values': x_hat_values,
            'y_hat_values': y_hat_values,
            'angle_bins': angle_bins, 
            'num_trials': len(angle_bins)
            }

