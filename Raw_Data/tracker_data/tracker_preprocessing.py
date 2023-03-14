import pandas as pd
from Raw_Data.tracker_data.read_trackers import read_tracker
from Raw_Data.tracker_data.filter_trials import filter_data, filter_short_trials
from Raw_Data.tracker_data.intepolate import data_interpolation
from Raw_Data.utils.timestamp_correction import timestamp_correction
from Raw_Data.utils.indices_of_interest import idx_of_return, idx_of_start, idx_of_back
import Raw_Data.configurations as cfg


def reset_index(data):
    for (_, df) in data:
        df.reset_index(inplace=True, drop=True)

    return data

def choose_relevent_parts(data, mode="all", ts_name='Hand_loc_Y'):
    # define filter function based on the chosen mode
    if mode == "all":
        fun = lambda x:x
    elif mode == "before":
        fun = lambda x:x.iloc[:idx_of_start(x[ts_name])]
    elif mode == "movement":
        fun = lambda x:x.iloc[idx_of_start(x[ts_name]):idx_of_back(x[ts_name])]
    elif mode == "reach":
        fun = lambda x:x.iloc[idx_of_start(x[ts_name]):idx_of_return(x[ts_name])]
    elif mode == "return":
        fun = lambda x:x.iloc[idx_of_return(x[ts_name]):idx_of_back(x[ts_name])]
    
    for i,(_, df) in enumerate(data): 
        data[i] = (data[i][0], fun(df))

    

    return data


def no_pupil_frame_filter(data):
    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1][(data[i][1]['right_pupil'] != -1) & (data[i][1]['left_pupil'] != -1)])

    return data


def timestamp_to_ms(data):
    for (_, df) in data:
        df['timestamp'] = timestamp_correction(df['timestamp'])

    return data


def timestamp_from_zero(data):
    for (_, df) in data:
        zero = df['timestamp'].iat[0]
        df['timestamp'] = df['timestamp'] - zero

    return data


def drop_extra(data):
    for (_, df) in data:
        df.drop(labels = cfg.to_drop, axis=1, inplace=True)
        
    return data

def cut_signal(data):
    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1][data[i][1]['timestamp'] > cfg.start_signal])

    return data

def tracker_preprocessing(subject_num):
    # read subject data
    data = read_tracker(subject_num)
    
    # reset dataframe index
    data = reset_index(data)
    
    # format timestamp in miliseconds
    data = timestamp_to_ms(data)
    
    # if the signal is pupil dimeteter, throw rows with -1 values
    if cfg.pathes.trial_mode.startswith('pupil'):
        data = no_pupil_frame_filter(data) # consider deleting few more frames before and after the blink
    
    # filter trials
    data = filter_data(data)
    
    # choose relevant part of the trial
    data = choose_relevent_parts(data, mode=cfg.tracker_cut_mode)
    
    # filter trials with to little amount of data left
    data = filter_short_trials(data)
    
    # drop extra columns
    if cfg.to_drop:
        data = drop_extra(data)
    
    # reset dataframe index
    data = reset_index(data)
    
    # start any trial's timestamp from zero
    data = timestamp_from_zero(data)

    # interpolate the data in order to hve even space between frames
    data = data_interpolation(data)
    
    # cut the begining of the signal
    if cfg.start_signal > 0:
        data = cut_signal(data)
        data = reset_index(data)
        data = timestamp_from_zero(data)
    
    return data 
