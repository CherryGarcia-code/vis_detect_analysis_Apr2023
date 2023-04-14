import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import json

####################################################################################################################

#%%
def get_mice_and_sessions(dict):
    mice_ids = list(dict.keys())
    all_sessions_by_mouse = []
    max_sessions = 0
    for mouse in mice_ids:
        sessions = dict[mouse].keys()
        all_sessions_by_mouse.append(list(sessions))
        if len(sessions)>max_sessions:
            max_sessions =  len(sessions)
    return mice_ids, all_sessions_by_mouse, max_sessions
####################################################################################################################

#%%
def get_early_lick_TFs (dict, mice_ids, all_sessions_by_mouse, max_sessions):
    mouse_sessions_df = pd.DataFrame(index=range(len(mice_ids)), columns=range(max_sessions))
    for idx, mouse in enumerate(mice_ids):
        second = 1*20
        sessions = all_sessions_by_mouse[idx]
        for session in sessions:
            trial_data = pd.DataFrame(dict[mouse][session]['behav_data']['trials_data_exp'])
            
            motion_onsets = dict[mouse][session]['Video']['MotionOnsetTimes']
            baseline_onsets = dict[mouse][session]['NI_events']['Baseline_ON']['rise_t']
            reaction_times = motion_onsets  - baseline_onsets

            early_lick_mask = trial_data.IsFA+trial_data.IsAbortWithFA
            early_lick_mask = early_lick_mask.astype(bool)

            early_trial_data = trial_data[early_lick_mask].reset_index(drop=True) 
            early_reaction_times = reaction_times[early_lick_mask]

            early_trial_data = early_trial_data[early_reaction_times>2].reset_index(drop = True)  
            early_reaction_times = early_reaction_times[early_reaction_times>2]
            n_early_trials  = len(early_trial_data)

            session_TF_windows = []
            for trial in range(n_early_trials):
                # early_reaction_times = np.around(early_reaction_times,decimals=2)
                trial_TF = early_trial_data['TF'][trial][early_trial_data['TF'][trial]!=0][::3] #For each trial within a session, take the TF sequence from where values are no longer 0. Then take the 3rd value each time, resulting in removal of triplicate values due to moving from 60Hz to 20Hz. 
                up_to_reaction_time_in_sec = np.around(np.arange(0,early_reaction_times[trial],0.05),decimals=2) #Create a vector of time values in seconds at 20Hz resolution for the duration between the initiation of grating and detection of motion leading to lick (from motion energy, not trial end)
                # len(up_to_reaction_time_in_sec)
                trial_TF_in_sec= np.arange(0,len(trial_TF))/20
                # len(trial_TF_in_sec)
                TF_window_start = np.squeeze(np.where(trial_TF_in_sec == up_to_reaction_time_in_sec[-2*second]))
                TF_window_end = TF_window_start+2*second
                trial_TF_window = trial_TF[TF_window_start:TF_window_end]
                session_TF_windows.append(trial_TF_window)    
            mouse_sessions_df.iloc[idx,sessions.index(session)] = pd.DataFrame(session_TF_windows)
    return mouse_sessions_df
####################################################################################################################
#%%
def show_line_with_errShade(df):
    n = len(df) #number of rows (observations/sessions) in the 2D df of time fragments out of 40 (corresponding to 2 seconds) vs sessions.
    se = stats.sem(df) # compute the column sem (i.e. the sem of all the nth time fragments across all sessions.)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1) #compute the 95% confidence interval with the se.
    all_session_mean  = df.mean()
    y1  = all_session_mean+h
    y2 = all_session_mean-h
    plt.plot(all_session_mean)
    plt.fill_between(x = np.arange(0,len(all_session_mean)), y1=y1,y2 = y2, alpha = 0.2)
    plt.xticks(np.arange(0,40,10),labels=np.arange(-2,0,0.5))
    plt.show()
####################################################################################################################
#%%
def get_behave_data(path_to_parent_folder, choose_outcome = 'all', return_dict_or_df = 'all', include_auto_reward_trials = False):

    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd

    print('path to parent folder:', path_to_parent_folder, ', choose outcome = ', choose_outcome, ', include auto reward trials = ', include_auto_reward_trials, ', return dict or df = ', return_dict_or_df) 
    path_to_parent_folder
    choose_outcome
    include_auto_reward_trials

    
   

    # 'X:/public/projects/BeJG_20230130_VisDetect/'
    data_path = Path(path_to_parent_folder) #Define the path to the global parent folder within Ceph shared lab space. 
    session_settings_paths = list(data_path.rglob("*session_settings.json")) # Get a list of all paths for the session settings. 
    per_session_trials_paths = list(data_path.rglob("*trials.json")) # Get a list of all paths for the trials conditions for all trials in a session.
    assert len(per_session_trials_paths) == len(session_settings_paths) # Check that the length of each list is the same (suggesting that each session settings file corresponds to a trials settings file).
    # Load the data from the json files for trial and session data:
    all_mouse_session_dict = {'mouse_id':[],'time_stamp':[],'session_trial_outcome':[],'reaction_time':[],'session_TF_vectors':[],
                              'baseline_TF':[],'change_size_TF':[],'change_time':[],'auto_reward':[]}
    for session_idx in np.arange(len(session_settings_paths)):
    # session_idx = 0
        try:
            with open(session_settings_paths[session_idx]) as session:
                session_settings_dict = json.load(session)
                session_settings = pd.Series(session_settings_dict) # per session settings relevant for all trials within a session.
            
            with open(per_session_trials_paths[session_idx]) as per_session_trials: # this is per session (json file includes all the trials that occured in that session)
                per_session_trials_dict= json.load(per_session_trials)
                per_session_trials_data = pd.Series(per_session_trials_dict) # output is a series where each element is a dictionary of values for a single trial.
        except:
            print('json files are empty for: ', session_settings_paths[session_idx] )
        # extract 
        # from session settings: temporalfreq, autorewd; 
        # From trials data: trialoutcome, reactiontimes, TF, stimT, Stim1TF (baseline freq), Stim2TF (change freq).

        # definition of columns for the data frame and their content in ditionary of lists with keys as column names:
        mouse_id = session_settings_paths[session_idx].name.split('_')[1]
        
        session_date = session_settings_paths[session_idx].name.split('_')[2]
        session_time = session_settings_paths[session_idx].name.split('_')[3]
        time_stamp = session_date+session_time
        baseline_temporal_freq = session_settings.temporalfreq
        auto_rewd = session_settings.autorewd

        for trial in np.arange(len(per_session_trials_data)):
            # trial = 0 #comment out after loop structuring
            all_mouse_session_dict['mouse_id'].append(mouse_id)
            all_mouse_session_dict['time_stamp'].append(time_stamp)
            trial_outcome = per_session_trials_data[trial]['trialoutcome']
            all_mouse_session_dict['session_trial_outcome'].append(trial_outcome) # a single string value of FA, Hit, Ref, Miss, Gray, abort
            if trial_outcome == 'Hit': trial_outcome = 'RT'
            all_mouse_session_dict['reaction_time'].append(per_session_trials_data[trial]['reactiontimes'][trial_outcome]) # an individual float value from a dictionary of reaction times for each one of the keys for the possible trial outcomes detailed above. 
            temporal_freq_vector = np.array(per_session_trials_data[trial]['TF']) # a list (turned into an array) of values for the temporal frequency from before grating started till the trial ended. 
            temporal_freq_filter = temporal_freq_vector > 0
            temporal_freq_vector = temporal_freq_vector[temporal_freq_filter]
            temporal_freq_vector = temporal_freq_vector[::3]
            all_mouse_session_dict['session_TF_vectors'].append(temporal_freq_vector)
            all_mouse_session_dict['baseline_TF'].append(per_session_trials_data[trial]['Stim1TF']) # the value of TF around which the baseline frequency revolves. 
            all_mouse_session_dict['change_size_TF'].append(per_session_trials_data[trial]['Stim2TF']) # the value of the change TF (change size out of 4,2,1.5,1.35,1.25)
            all_mouse_session_dict['change_time'].append(per_session_trials_data[trial]['stimT'])
            all_mouse_session_dict['auto_reward'].append(auto_rewd)
    if return_dict_or_df != 'dict':
        master_df = pd.DataFrame(all_mouse_session_dict)
        try:
            master_df['time_stamp'] = pd.to_datetime(master_df['time_stamp'],format='%Y%m%d%H%M%S')
        except:
            print('time stamp field of a file requires modification to convert to datetime')
        master_df = master_df[(master_df['change_size_TF']>=1.25) & (master_df['change_size_TF']<15)]
        if choose_outcome != 'all':
            master_df = master_df[master_df['session_trial_outcome']==choose_outcome]
        if include_auto_reward_trials == False:
            master_df = master_df[master_df['auto_reward'] == 0]

        master_df.reset_index(drop=True,inplace=True)

        if return_dict_or_df == 'df': return master_df 
        if return_dict_or_df == 'all': return all_mouse_session_dict, master_df
        
    # if return_dict_or_df := 'df': return master_df
    
    return all_mouse_session_dict
#####################################################################################################################

# %%
