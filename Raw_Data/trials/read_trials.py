import pandas as pd
import numpy as np
from os import listdir

from Raw_Data.utils.utils import number_to_string
import Raw_Data.configurations as cfg
import pathes


def path_resolver(subject_num):
    path = pathes.raw_data_path + cfg.participant_dir_name + number_to_string(subject_num) + '/Unreal/'
    path += listdir(path)[0] + '/UsedPlan/'

    suffix = [f for f in listdir(path) if f.startswith(cfg.trials_file_name_prefix)]
        
    path = path + '/' + suffix[0]
    
    return path


def filter_trials_file_question(df):
    # take only real trials rows
    # filter by question
    df = df[~df.iloc[:, cfg.trial_relevant_trials_filter_col].isin(cfg.trial_unrelevant_codes)]
    
    
    # take only id and relevant columns
    trial_index_idx = df.columns.get_loc(cfg.trial_index_col_name)
    df = df.iloc[:, [trial_index_idx] + cfg.trial_condition_col_idx]
    
    return df
    
def label_trials_file_one_column(df, col_id, tranfrom_dic=cfg.trial_labels_dic):
    df.iloc[:,col_id].replace(tranfrom_dic, inplace=True)
    return df

def label_trials_file(df, col_id, tranfrom_dic=cfg.trial_labels_dic):
    trial_cfg = df.iloc[:,col_id: col_id + len(cfg.trial_relevant_cols)]
    trial_cfg = [tuple(x.to_list()) for _, x in trial_cfg.iterrows()]
    trial_labels = [tranfrom_dic[x] for x in trial_cfg]
    df.iloc[:, col_id] = trial_labels 
    df = df.iloc[:, [0,col_id]]
    return df

def add_index(trials_csv):
    trials_csv[cfg.trial_index_col_name] = trials_csv.index


def read_trials(subject_num):
    # resolve path of trials file
    path = path_resolver(subject_num)
    
    # read trials data
    data = pd.read_csv(path, header=None)

    # add a index coloumn:
    add_index(data)

    # filter trials data
    data = filter_trials_file_question(data)

    # reset index
    data.reset_index(inplace=True, drop=True)

    return data