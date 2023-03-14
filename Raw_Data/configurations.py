import os
import pathes
# pathes
# generate path of to data directories, without dependency on out locaation
# relational data directories location should be fixed


# tracker data preprocessing configurations
tracker_cut_mode = "all"
tracker_file_name = "TrackersOutputData.csv"
tracker_file_name_prefix = 'TrackersOutput'
relevant_rows_filter = [(3, "Room"), (4, "NoBlockView")]
unrelevant_trials = (0, 80, 666, 777, 9992)
tracker_idx_col = 'idx'
tracker_trialconf_col = 'BlockNumber'
tracker_relevant_data = [(65, tracker_idx_col), (6, 'timestamp'), (29, 'Hand_loc_Z'), (49, 'right_pupil'),
                         (50, 'left_pupil'),
                         ]
# tracker_relevant_data = [(65, tracker_idx_col), (6, 'timestamp'), (27, 'Hand_loc_X'), (28, 'Hand_loc_Y'), (29, 'Hand_loc_Z'),]
to_drop = 'Hand_loc_Z'
start_signal = 650
# for hand: tracker_relevant_data = [(1, tracker_idx_col), (6, 'timestamp'), (27, 'Hand_loc_X'), (28, 'Hand_loc_Y'), (29, 'Hand_loc_Z'),]
# for pupil: tracker_relevant_data = [(1, tracker_idx_col), (6, 'timestamp'), (29, 'Hand_loc_Z'), (49, 'right_pupil'), (50, 'left_pupil'),]
    
tracker_relevant_data_cols = [x[0] for x in tracker_relevant_data]
tracker_relevant_data_names = [x[1] for x in tracker_relevant_data]



# answers preprocessing configurations
answer_relevant_cols = ['QuestionResult']
answers_index = '#Index'
relevant_question = 1
answers_file_name = "Answers.csv"
answers_file_name_prefix = "Answers"
answer_question_col_name = "QuestionID"

# trials preprocessing configurations
trial_condition_col_idx = [0,1]
trial_relevant_trials_filter_col = 1
trial_index_col_name = "#trial number"
trial_relevant_cols = ['SensoMotoric Delay', 'angleChange']
trial_unrelevant_codes = (0, 80, 666, 777, 9992)
filter_training = -1
trials_file_name = ['trials.csv']
trials_file_name_prefix = 'trial'
trial_labels_dic = {(0,0): 0, (0.05,0): 1, (0.1,0):2, (0.15,0):3, (0,0.2126):4, (0,0.2867):5, (0,0.364):6}
part_of_movement = pathes.trial_mode

# for asaf:trial_labels_dic = {(0,0): 0, (0.05,0): 1, (0.1,0):2, (0.15,0):3, (0,0.2126):4, (0,0.2867):5, (0,0.364):6}
# for yoni:trial_labels_dic = {(0): 0, (0.033):1, (0.044):2, (0.066):3, (0.099):4, (0.154):5, (0.231):6, (0.352):7}
# for ophir:trial_labels_dic = {(0,0): 0, (0.05,0): 1, (0.1,0):2, (0.15,0):3, (0.2,0):4}


# reading files configuration
participant_dir_name = 'sub_'
numbers_mode = 3




# filters
filter_column_of_interest = 'Hand_loc_Z'
filter_time_short = 600
filter_movement_short = 0.15
filter_movement_long = 0.8
filter_expected_low = 0.75
filter_expected_high = 0.94
min_reaching = .1
hesitation_threshold = 100
too_short_trial = 10

# interpolation
rate_hz = 11


# padding
padding_value = -10