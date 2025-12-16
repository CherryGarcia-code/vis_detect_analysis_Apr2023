from glob import glob
import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import statistics
from scipy.signal import savgol_filter
import scipy.stats as stats
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns

Hz = 1
sampling_freq = 100 * Hz
second = 100
minute = 60 * second
smooth_poly = 4

def load_csv_data(filepath):
    return pd.read_csv(filepath)

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)
    
def save_as_pickle(file_path, data):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(file_path):
    with open(file_path, "rb") as file:
        preprocessed_data  =  pickle.load(file)
    return preprocessed_data


def concatenate_pickles(mouse_dir, output_filename='big_session_data.pkl'):
    """
    Concatenates all .pkl files in the specified directory that have 'protocol' column value 4
    into a single dataframe and saves it to a specified output file.

    Parameters:
    mouse_dir (str): The directory containing the preprocessed .pkl files.
    output_filename (str): The filename for the output concatenated dataframe. Default is 'big_session_data.pkl'.

    Returns:
    pd.DataFrame: The concatenated dataframe.
    """
    directory = f'{mouse_dir}/preprocessed' if os.path.exists(f'{mouse_dir}/preprocessed') else mouse_dir

    # List to hold each dataframe
    dataframes = []

    # Get all .pkl files and sort them by timestamp in filenames
    pkl_files = sorted([filename for filename in os.listdir(directory) if filename.endswith('.pkl')])

    # Iterate over sorted files in the directory
    for filename in pkl_files:
        filepath = os.path.join(directory, filename)
        df = pd.read_pickle(filepath)

        # Debug: Check if df is None
        if df is None:
            print(f"Warning: {filepath} returned None")
            continue

        # Debug: Check if 'protocol' column exists
        if 'protocol' not in df.columns:
            print(f"Warning: {filepath} does not contain 'protocol' column")
            continue

        # Check if 'protocol' column value is '4'
        if df['protocol'].eq(4).all():  # Ensure all values in 'protocol' column are '4'
            dataframes.append(df)

    # Concatenate all dataframes
    if dataframes:
        big_dataframe = pd.concat(dataframes, ignore_index=True)
        # Save the concatenated dataframe to a new pickle file
        big_dataframe.to_pickle(f'{directory}/{output_filename}')
    else:
        print("No valid dataframes found.")
        return None

    return big_dataframe

def extract_timestamp_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 2:
        return parts[2]
    return None

def get_all_timestamps(mouse_dir):
    timestamps = set()
    for filename in os.listdir(mouse_dir):
        if filename.endswith('trials.json'):
            timestamp = extract_timestamp_from_filename(filename)
            if timestamp:
                timestamps.add(timestamp)
    return list(timestamps)

def format_timestamp_for_photom_files(timestamp):
        return re.sub(r'(\d{4})(\d{2})(\d{2})', r'\1-\2-\3', timestamp)

def get_session_files(mouse_dir, timestamp=False):
    # Glob all session settings and trials files in the directory
    session_settings_files = glob(os.path.join(mouse_dir, '*__session_settings.json'))
    trials_files = glob(os.path.join(mouse_dir, '*__trials.json'))
    photom_files = glob(os.path.join(mouse_dir, '*__photom_*.csv'))
    photom_io_files = glob(os.path.join(mouse_dir, '*__photom_IO_*.csv'))
    photom_files = [file for file in photom_files if file not in photom_io_files]
    session_settings_files = glob(os.path.join(mouse_dir, '*__session_settings.json'))

    
    # Filter files by timestamp if a specific timestamp is provided
    if timestamp:
        # Format the timestamp for photom and photom_io files
        formatted_timestamp = format_timestamp_for_photom_files(timestamp)
        print(formatted_timestamp)
        # Assuming the timestamp is part of the file's name, adjust the condition to match your file naming convention
        session_settings_files = [file for file in session_settings_files if timestamp in os.path.basename(file)]
        trials_files = [file for file in trials_files if timestamp in os.path.basename(file)]
        photom_files = [file for file in photom_files if formatted_timestamp in os.path.basename(file)]
        photom_io_files = [file for file in photom_io_files if formatted_timestamp in os.path.basename(file)]

    print('number of files: session_settings: ', len(session_settings_files), '\n trials: ', len(trials_files), '\n photom: ', len(photom_files), '\n photom_io: ', len(photom_io_files))
    
    return photom_files, photom_io_files, session_settings_files, trials_files

def parse_timestamp(date_str, time_str):
    # Combine date and time strings into a single datetime string
    datetime_str = f"{date_str}{time_str}"
    # Specify the format corresponding to 'YYYYMMDDHHMMSS'
    datetime_format = "%Y%m%d%H%M%S"
    # Parse the datetime string into a datetime object
    return datetime.strptime(datetime_str, datetime_format)

def categorize_change_size(value):
    if value in [2, 4]:
        return 'big'
    elif value in [1.25, 1.35, 1.5]:
        return 'small'
    elif value == 1:
        return 'no_change'
    else:
        return 'unknown'  # Optional: handle any other unexpected value


def calculate_dff(trial_df, baseline_timestamp, session_zscored = True):  ######## change session_zscored to 'False' in case you want to z score to iti period for each trial. 
    # Select the baseline period
    baseline_period = trial_df[trial_df['SystemTimestamp'] <= baseline_timestamp]
    
    # Calculate the mean signal during the baseline period for each photometry signal
    baseline_means = {
        column: baseline_period[column].mean() for column in trial_df.columns if 'clean_signal_dff' in column
    }
    baseline_stds = {
        column: baseline_period[column].std() for column in trial_df.columns if 'clean_signal_dff' in column
    }
    
    # Calculate ΔF/F for each signal
    trial_dff = pd.DataFrame()
    trial_df_copy = trial_df.copy()  # Create a copy to avoid modifying the original DataFrame
    for (signal, baseline_mean), (_, baseline_std) in zip(baseline_means.items(), baseline_stds.items()):
        
        dff_column_name = f'{signal}'
        # trial_dff[dff_column_name] = (trial_df_copy[signal] - baseline_mean) / baseline_mean # Calculate ΔF/F with a separate baseline according to each trial's ITI period. 
        trial_dff['SystemTimestamp'] = trial_df_copy['SystemTimestamp']
        # Calculate the z-score relative to the baseline
        if session_zscored == False:
            if baseline_std != 0:  # To avoid division by zero
                trial_dff[f'zscored_{dff_column_name}'] = (trial_df_copy[signal] - baseline_mean) / baseline_std
            else:
                trial_dff[f'zscored_{dff_column_name}'] = 0
        else:
             trial_dff[f'{dff_column_name}'] = trial_df_copy[f'{dff_column_name}']
        
    return trial_dff, baseline_means, baseline_stds

def get_window_indices(trial, window_start, window_end, sampling_rate):
    event_time = trial['reaction_times_from_reference_start']
    start_index = int((event_time + window_start) * sampling_rate)
    end_index = int((event_time + window_end) * sampling_rate)
    return start_index, end_index

def find_nearest(timestamps, value):
        # Find the timestamp in the DataFrame that is closest to the given value
        return timestamps.iloc[(timestamps - value).abs().argsort().iloc[0]]


def extract_photometry_signals(photom_data, start_times, reaction_times):
        signals = []  # This will store the segments for each trial
        
        for start, duration in zip(start_times, reaction_times):
            end = start + duration + 2.0 # Calculate the end timestamp for the trial
            # Extract rows where the timestamp is within the start and end range
            signal = photom_data[(photom_data['SystemTimestamp'] >= start) & (photom_data['SystemTimestamp'] < end)]
            signals.append(signal)
        
        return signals

def extractSessionMetadata(session_settings, base_filename):
        # Extract the mouse ID from the session settings
        mouse_id = session_settings['token'].split('_')[1]

        
        # Split the base file name by underscore
        parts = base_filename.split('_')
        # The date is the third element from the end before the file type
        session_date = parts[2]
        # The time is the second element from the end before the file type
        session_time = parts[3]

        # Combine date and time strings to datetime object
        session_timestamp = parse_timestamp(session_date, session_time)

        ### Extract data of interest from session settings:
        auto_rewd = session_settings['autorewd']
        
        # Extract the punishment value from the session settings
        punishment_value = session_settings['punishearly']
        if punishment_value == "End trial on Stim1 lick":
            punishment = 'end_trial'
        elif punishment_value == "Air-puff Stim1 lick":
            punishment = 'air_puff'
        elif punishment_value == "Ignore Stim1 lick":
            punishment = 'ignore_licks'
        elif punishment_value == "Time-Out Stim1 lick":
            punishment = 'timeout'
        else:
            punishment = 'no_punishment'
             
        
        # Extract the protocol value from the session settings
        if session_settings['hazardtype'] == 'split block':
            protocol = 5
        elif session_settings['pprobe0'] == 0.2:
            protocol = 4
        elif session_settings['pprobe0'] == 0.5:
            protocol = 3
        elif session_settings['Trewdavailable'] == 0.5:
            protocol = 1
        else:
            protocol = 2

        return mouse_id, session_date,  session_time,  session_timestamp, auto_rewd, punishment, protocol




def extractTrialsData (trials_data):
        # for each trial in the current session, Extract data of interest from trials data:
        iti_values = []
        reaction_times = []  
        reaction_times_from_reference_start = []
        outcomes = []
        change_times = []
        change_sizes_TF = []
        TF_vectors = []
        laser_states = []
    

        for trial in trials_data:
            iti = trial['stimD']
            iti_values.append(iti)

            trial_outcome = trial['trialoutcome']
            outcomes.append(trial_outcome)

            if trial_outcome == 'Hit': trial_outcome = 'RT'

            reaction_time = trial['reactiontimes'][trial_outcome] 
            reaction_times.append(reaction_time)

            change_time = trial['stimT']
            change_times.append(change_time)

            reaction_time_from_reference_start = reaction_time + iti + change_time if (trial_outcome == 'RT' or trial_outcome == 'Miss') else reaction_time+iti
            reaction_times_from_reference_start.append(reaction_time_from_reference_start)

            

            TF_vector = np.array(trial['TF'])
            TF_vector = TF_vector[TF_vector > 0]
            TF_vector = TF_vector[::3]
            TF_vectors.append(TF_vector)
            
            change_size_TF = trial['Stim2TF']
            change_sizes_TF.append(change_size_TF)
            
            try:
                laser_state = trial['LaserOn']
                laser_states.append(laser_state)
            except:
                laser_states.append(0)
        return iti_values, reaction_times, reaction_times_from_reference_start, outcomes, change_times, change_sizes_TF, TF_vectors, laser_states


# UPDATED function which checks whether there are files with no baseline signals and then just returns None if so:
def process_session(session_data):
    photometry_signals = session_data['photometry_signals']
    baseline_on_timestamps = session_data.get('baseline_on_timestamps', [])  # Use .get with a default empty list
    
    # Check if baseline_on_timestamps is empty and handle it
    if not baseline_on_timestamps:
        print("No baseline on timestamps available.")
        # Handle the lack of baseline timestamps according to your requirements
        # For example, return None or an empty list to indicate no data
        return None, None, None
    
    all_trials_dff = []  # Initialization of the list
    all_trials_baseline_means = []
    all_trials_baseline_stds = []
    for i, trial_df in enumerate(photometry_signals):
        # Ensure that there is a corresponding baseline timestamp for each trial
        if i < len(baseline_on_timestamps):
            baseline_timestamp = baseline_on_timestamps[i]
            trial_dff, baseline_mean, baseline_std = calculate_dff(trial_df, baseline_timestamp)
            all_trials_dff.append(trial_dff.reset_index(drop=True))
            all_trials_baseline_means.append(baseline_mean)
            all_trials_baseline_stds.append(baseline_std)
        else:
            print(f"No baseline timestamp for trial index {i}.")
            # Handle the lack of a baseline timestamp for the trial
            # For example, skip this trial, or append None or a placeholder
            all_trials_dff.append(None)
            all_trials_baseline_means.append(None)
            all_trials_baseline_stds.append(None)

    return all_trials_dff, all_trials_baseline_means, all_trials_baseline_stds

def process_session_data(photom_files, photom_io_files, session_settings_files, trials_files):
    # Example for one session
    print('******************   NEW   SESSION   ******************')
    
    photom_df = load_csv_data(photom_files[0])
    photom_io_df = load_csv_data(photom_io_files[0])
    session_settings = load_json_data(session_settings_files[0])
    trials_data = load_json_data(trials_files[0])
   
    # Extract the base file name without the directory
    base_filename = os.path.basename(session_settings_files[0])

    
    
    mouse_id, session_date,  session_time,  session_timestamp, auto_rewd, punishment, protocol = extractSessionMetadata(session_settings, base_filename)
    iti_values, reaction_times, reaction_times_from_reference_start, outcomes, change_times, change_sizes_TF, TF_vectors, laser_states = extractTrialsData(trials_data)
    

    photom_df_preprocessed = get_signal(photom_df,session_id = session_date+session_time,session_zscored = True, plot=False) ### change here to determine whether zscore should be applied for each session or each iti period per trial. 


    print('len iti_values:', len(iti_values))
    baseline_on_df = photom_io_df[photom_io_df['DigitalIOName'] == 'Input0']
    baseline_on_timestamps = baseline_on_df['SystemTimestamp'].tolist()
    lick_on_df = photom_io_df[photom_io_df['DigitalIOName'] == 'Input1']
    lick_on_timestamps = lick_on_df['SystemTimestamp'].tolist()
    print('len baseline_on_timestamps:', len(baseline_on_timestamps))

    if len(iti_values) != len(baseline_on_timestamps):
        return None
    reference_start_actual = [(a - b) for a, b in zip(baseline_on_timestamps, iti_values)]
    change_times_actual = [(a + b) for a, b in zip(baseline_on_timestamps, change_times)]
    
    
    
    
    # Find the closest timestamp for each value in values_list
    values_list = reference_start_actual 
    
    # Using list comprehension to find the closest timestamp for each value in values_list
    if photom_df_preprocessed is not None:
        reference_start_photometry= [find_nearest(photom_df_preprocessed['SystemTimestamp'], value) for value in values_list]

        reference_start_error = [(a - b) for a, b in zip(reference_start_photometry, reference_start_actual)] 
        
        change_time_Timestamps = [find_nearest(photom_df_preprocessed['SystemTimestamp'], value) for value in change_times_actual]
        
        reaction_times_actual = [(a + b) for a, b in zip(values_list, reaction_times_from_reference_start)]
        photometry_signals = extract_photometry_signals(photom_df_preprocessed, reference_start_photometry, reaction_times_from_reference_start)
    
        reaction_time_Timestamps = [find_nearest(photom_df_preprocessed['SystemTimestamp'], timestamp) for timestamp in reaction_times_actual]  


    print('mouse_id is: ',mouse_id)
    print('session_date is: ', session_date)
    print('session_time is: ', session_time)
    print('session timestamp is: ',session_timestamp)
    print('len iti_values:', len(iti_values))
    print('auto rewd is: ',auto_rewd)
    print('punishment is: ',punishment)
    print('protocol is: ',protocol)
    print('unique outcomes are: ',np.unique(outcomes))
    print('unique laser states are: ',np.unique(laser_states))
    # print('iti values are: ',iti_values)
    # print('baseline on timestamps are: ', baseline_on_timestamps)
    # print('reference start actual is: ', reference_start_actual)
    # print('reference start photometry is: ', reference_start_photometry)
    # print('error between reference start photometry and actual is: ',reference_start_error)

    processed_data = {
        'mouse_id': mouse_id,
        'session_date': session_date,
        'session_time': session_time,
        'session_timestamp': session_timestamp,
        'auto_rewd': auto_rewd,
        'punishment': punishment,
        'protocol': protocol,
        'iti_values': iti_values,
        'baseline_on_timestamps': baseline_on_timestamps,
        'reference_start_actual': reference_start_actual,
        'reference_start_photometry': reference_start_photometry,
        'reference_start_error': reference_start_error,
        'change_times': change_times,
        'change_sizes_TF': change_sizes_TF,
        'outcomes': outcomes,
        'reaction_times': reaction_times,
        'reaction_times_from_reference_start': reaction_times_from_reference_start,
        'reaction_time_Timestamps': reaction_time_Timestamps,
        'change_time_Timestamps': change_time_Timestamps,
        'TF_vectors': TF_vectors,     
        'photometry_signals': photometry_signals,
        'laser_states': laser_states        
    }

    processed_data['dff_data'], processed_data['baseline_means'], processed_data['baseline_stds'] = process_session(processed_data)
    
    session_data_df = pd.DataFrame(processed_data)

    session_data_df['change_category'] = session_data_df['change_sizes_TF'].apply(categorize_change_size)

    return session_data_df


    

def flatten_nested_df(all_data_df):

    # Step 1: Expand the nested DataFrame structure into a list of DataFrames
    dataframes_list = []
    for subject_id, series in all_data_df.items():
        for session_id, session_df in series.items():
            if session_df is not None:
                # Assign new index levels to the session_df for proper concatenation
                # session_df = session_df.assign(subject_id=subject_id, session_id=session_id)
                dataframes_list.append(session_df)

    # Step 2: Concatenate these DataFrames into a single DataFrame with a MultiIndex
    concat_df = pd.concat(dataframes_list)
    return concat_df



def filter_and_pad_data(df, change_sizes, threshold):
    """
    Filters and pads the data according to the specified change sizes and performance threshold.

    Parameters:
    - df: The DataFrame containing the data.
    - change_sizes: A list of desired change sizes to consider.
    - threshold: The performance threshold to determine passing sessions.

    Returns:
    - A DataFrame with the filtered and padded data.
    """
    subsetted_sessions = {}
    max_passing_sessions = 0

    # Determine the maximum number of passing sessions for any subject
    for subject in df.columns:
        if subject.endswith('_performance'):
            for change_size in change_sizes:
                passing_sessions_count = sum(
                    performance_data.get(change_size, (0, 0))[0] > threshold
                    for performance_data in df[subject] if performance_data is not None
                )
                max_passing_sessions = max(max_passing_sessions, passing_sessions_count)

    # Collect passing sessions and apply padding
    for subject in df.columns:
        if not subject.endswith('_performance'):
            continue

        subject_name = subject.replace('_performance', '')
        for change_size in change_sizes:
            selected_sessions = [
                df[subject_name][i] for i, performance_data in enumerate(df[subject])
                if performance_data is not None and performance_data.get(change_size, (0, 0))[0] > threshold
            ]

            padding_needed = max_passing_sessions - len(selected_sessions)
            selected_sessions.extend([None] * padding_needed)
            subsetted_sessions[f"{subject_name}"] = selected_sessions

    return pd.DataFrame.from_dict(subsetted_sessions, orient='index').transpose()


def flatten_nested_df(all_data_df):

    # Step 1: Expand the nested DataFrame structure into a list of DataFrames
    dataframes_list = []
    for subject_id, series in all_data_df.items():
        for session_id, session_df in series.items():
            if session_df is not None:
                # Assign new index levels to the session_df for proper concatenation
                # session_df = session_df.assign(subject_id=subject_id, session_id=session_id)
                dataframes_list.append(session_df)

    # Step 2: Concatenate these DataFrames into a single DataFrame with a MultiIndex
    concat_df = pd.concat(dataframes_list)
    return concat_df

def calculate_snr(signal, noise):
    # Calculate the power of the signal and noise
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    # Calculate SNR in decibels
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def get_signal(df, session_id, smooth_poly = 4, session_zscored = True,plot = True, save_plots = False,snr_threshold=5,
               output_dir = 'D:\python_analysis\git_repos\vis_detect_analysis_Apr2023\photom_plots'):
    
    clean_signal_df = pd.DataFrame()

    data = df.copy()
    
    # Assign ROI columns based on presence of additional ROIs for lateral hemispheres
    dms_rois = ['G0', 'G2']  # Dorsomedial striatum
    vls_rois = ['G4', 'G5']  # Ventrolateral striatum, if present
    
    # Check if VLS data is present
    rois = dms_rois + vls_rois if 'G4' in data.columns and 'G5' in data.columns else dms_rois
        

    # Process each ROI and add the results as new columns
    for roi in rois:
        hemisphere = 'left' if '0' in roi or '4' in roi else 'right'
        region = 'DMS' if roi in dms_rois else 'VLS'
        
        # Extract the timestamps for isosbestic (1) and signal (2) data
        iso_timestamps = data.loc[data['LedState'] == 1, 'SystemTimestamp'].to_numpy()
        sig_timestamps = data.loc[data['LedState'] == 2, 'SystemTimestamp'].to_numpy()

        # Extract and process the isosbestic and signal data
        iso_data = data.loc[data['LedState'] == 1, roi].to_numpy()
        sig_data = data.loc[data['LedState'] == 2, roi].to_numpy()
        
        # print(type(iso_data), type(sig_data))
        
       
        # Trim the beginning of the arrays to make sure the artifact of turning on the LED is not included
        trim_samples = 10 * second # Number of samples to trim at 100 Hz per channel is 300 for 3 seconds. 
        
        iso_data = iso_data[trim_samples:]
        sig_data = sig_data[trim_samples:]
        iso_timestamps = iso_timestamps[trim_samples:]
        sig_timestamps = sig_timestamps[trim_samples:]

        # Ensure that iso_data and sig_data have the same length
        min_length = min(len(iso_data), len(sig_data))
        iso_data = iso_data[:min_length]
        sig_data = sig_data[:min_length]
        iso_timestamps = iso_timestamps[:min_length]
        sig_timestamps = sig_timestamps[:min_length]

        # Fit and evaluate the linear model for isosbestic points
        iso_coef = np.polyfit(iso_data, sig_data, deg=1)
        iso_fitted = np.polyval(iso_coef, iso_data)

        # Smoothing the iso and signal data
        # iso_smooth = savgol_filter(iso_fitted, window_length=91, polyorder=smooth_poly)
        # sig_smooth = savgol_filter(sig_data, window_length=41, polyorder=smooth_poly+1)
        iso_smooth = savgol_filter(iso_fitted, window_length=90, polyorder=smooth_poly-1)
        sig_smooth = savgol_filter(sig_data, window_length=40, polyorder=smooth_poly-2)

        # Subtract the iso_smooth from sig_smooth to remove motion artifacts
        sig_smooth_clean = (sig_smooth - iso_smooth)
        sig_smooth_clean_dff = (sig_smooth_clean/iso_smooth)

        # Calculate the noise (residuals after fitting)
        noise = sig_data - iso_fitted
        
        # Calculate SNR
        snr = calculate_snr(sig_smooth_clean, noise)

        # Check if the SNR is above the threshold
        # if snr < snr_threshold:
            # print(f"Session {session_id}, ROI {roi} discarded due to low SNR: {snr:.2f} dB")
            # continue  # Skip the rest of the loop and do not add the data to clean_signal_df

        if sig_smooth_clean.min() < 0:
            sig_smooth_clean = sig_smooth_clean - sig_smooth_clean.min()
        if sig_smooth_clean_dff.min() < 0:
            sig_smooth_clean_dff = sig_smooth_clean_dff - sig_smooth_clean_dff.min()

        
        # Store the cleaned signal and corresponding timestamps in the dataframe
        clean_signal_df['SystemTimestamp'] = sig_timestamps
        clean_signal_df[f'{roi}_clean_signal'] = sig_smooth_clean
        clean_signal_df[f'{roi}_clean_signal_dff'] = sig_smooth_clean_dff
        if session_zscored == True:
            clean_signal_df[f'zscored_{roi}_clean_signal_dff'] = (sig_smooth_clean_dff - sig_smooth_clean_dff.mean())/sig_smooth_clean_dff.std()
        # After processing all ROIs
        # if clean_signal_df.empty:
        #     print(f"All ROIs for session {session_id} were discarded due to low SNR.")
        #     # Handle the case where clean_signal_df is empty
        #     # For example, you can return None or raise an exception
        #     return clean_signal_df
        

    # print(clean_signal_df.head())
        
                
        if plot == True:
            # photom_window = np.arange(0, 60 * minute)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            
            # Convert timestamps to seconds; assuming they are initially in sample indices
            iso_timestamps_seconds = iso_timestamps / sampling_freq
            sig_timestamps_seconds = sig_timestamps / sampling_freq
            data_length = len(iso_data)
            time_vector = np.arange(0, data_length / sampling_freq, 1/sampling_freq)
        
            # Plot the original isosbestic data and the original signal data
            ax[0].plot(time_vector, iso_data, label='Original Isosbestic', color='gray', alpha=0.7, linewidth=0.5)
            ax[0].plot(time_vector, sig_data, label='Original Signal', color='black', alpha=0.7, linewidth=0.5)
            ax[0].set_title(f'{hemisphere.capitalize()} {region} - Original Data')
            ax[0].legend()
            
            # Plot the smoothed isosbestic data and the smoothed signal data
            ax[1].plot(time_vector, iso_smooth, label='Smoothed Isosbestic', color='gray', linewidth=0.5)
            ax[1].plot(time_vector, sig_smooth, label='Smoothed Signal', color='black', linewidth=0.5)
            ax[1].set_title('Smoothed Data')
            ax[1].legend()
            
            # Plot the cleaned smoothed signal data after subtracting the isosbestic
            ax[2].plot(time_vector, sig_smooth_clean, label='Cleaned Signal', color='black', linewidth=0.5)
            ax[2].set_title('Cleaned Signal Data')
            ax[2].legend()
            
            # Plot the signal DFF
            ax[3].plot(time_vector, sig_smooth_clean_dff, label='Cleaned Signal_dff', color='black', linewidth=0.5, alpha=0.7)
            ax[3].set_title('Signal-to-Noise')
            ax[3].legend()
            
            # Set common labels
            plt.xlabel('Time (seconds)')
            plt.setp(ax, ylabel='Signal dff')
            
            # Adjust layout and show plot   
            plt.tight_layout()
            plt.show()
        
            if save_plots:
                fig.savefig(os.path.join(output_dir, f'{session_id}_{hemisphere}_{region}.png'))
                


              
       
    return clean_signal_df


def extract_signal_window_from_trial_df(df, event_timestamp, window_size=2.0, fixed_window_size=401):
    # Get the nested DataFrame for the trial
    trial_data = df['dff_data']
    # Calculate the window start and end times
    window_start = event_timestamp - window_size
    window_end = event_timestamp + window_size
    # Filter the trial data to the window
    window_data = trial_data[(trial_data['SystemTimestamp'] >= window_start) &
                             (trial_data['SystemTimestamp'] <= window_end)]
    window_data['mouse_id'] = df['mouse_id']
    window_data['session_date'] = df['session_date']
    window_data['change_sizes_TF'] = df['change_sizes_TF']
    window_data['change_category'] = df['change_category']
    # Pad the window data to the fixed size
    padding_size = fixed_window_size - len(window_data)
    if padding_size > 0:
        padding_index = np.linspace(-window_size, window_size, padding_size, endpoint=True)
        padding_df = pd.DataFrame(index=padding_index, columns=window_data.columns)
        # Drop all-NA columns before concatenation
        padding_df = padding_df.dropna(axis=1, how='all')
        window_data = pd.concat([window_data, padding_df]).sort_index().ffill().bfill().tail(fixed_window_size)
    window_data = window_data.set_index(np.linspace(-window_size, window_size, fixed_window_size, endpoint=True))
    
    return window_data

def extract_photom_windows_from_session_s(session_data, behave_event):
    # If the focus is on hit trials:
    if behave_event == 'hit':
        hits = session_data[session_data['outcomes'] == 'Hit']
        hit_signals = hits.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        change_signals = hits.apply(lambda row: extract_signal_window_from_trial_df(row, row['change_time_Timestamps']), axis=1).tolist()
        baseline_signals = hits.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        return hit_signals, change_signals, baseline_signals
    
    # If the focus is on miss trials:
    elif behave_event == 'miss':
        misses = session_data[session_data['outcomes'] == 'Miss']
        miss_signals = misses.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        change_signals = misses.apply(lambda row: extract_signal_window_from_trial_df(row, row['change_time_Timestamps']), axis=1).tolist()
        baseline_signals = misses.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        return miss_signals, change_signals, baseline_signals
    
    # If the focus is on change presentation (for relevant behavioral outcomes (not FA or abort where change is not encountered))
    elif behave_event == 'change':
        # Take trials with change presentation (remove FAs and aborts since change is not reached)
        no_FA_aborts = session_data[~((session_data['outcomes'] == 'FA') | (session_data['outcomes'] == 'abort'))]
        change_signals = no_FA_aborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['change_time_Timestamps']), axis=1).tolist()
        return change_signals, None
    
    # If the focus is on FA trials:
    elif behave_event == 'FA':
        # To take all FA trials
        FAs = session_data[session_data['outcomes'] == 'FA']
        # To make an early FA df without early FAs (under 2 seconds from baseline presentation)
        early_FAs = FAs[FAs['reaction_times'] <= 2]
        late_FAs = FAs[~((FAs['outcomes'] == 'FA') & (FAs['reaction_times'] < 2))]
        early_FA_signals = early_FAs.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        late_FA_signals = late_FAs.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        early_FA_baseline_signals = early_FAs.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        late_FA_baseline_signals = late_FAs.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        return early_FA_signals, late_FA_signals, early_FA_baseline_signals, late_FA_baseline_signals

    # If the focus is on abort trials:
    elif behave_event == 'abort':
        aborts = session_data[session_data['outcomes'] == 'abort']
        early_aborts = aborts[aborts['reaction_times'] <= 2]
        late_aborts = aborts[aborts['reaction_times'] > 2]
        early_aborts_signals = early_aborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        late_aborts_signals = late_aborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
        early_aborts_baseline_signals = early_aborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']) , axis=1).tolist()
        late_aborts_baseline_signals = late_aborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        return early_aborts_signals, late_aborts_signals, early_aborts_baseline_signals, late_aborts_baseline_signals
     

    # If the focus is on baseline presentation (for all behavioral outcomes)
    elif behave_event == 'baseline':
        no_earlyFAs_orAborts = session_data[~(((session_data['outcomes'] == 'FA')|(session_data['outcomes'] == 'abort')) & (session_data['reaction_times'] < 2))]
        baseline_signals = no_earlyFAs_orAborts.apply(lambda row: extract_signal_window_from_trial_df(row, row['baseline_on_timestamps']), axis=1).tolist()
        return baseline_signals, None

    

    else:
        raise ValueError(f"Unexpected behave_event: {behave_event}")
    

def melt_signals(signals, behave_event):
    # Drop columns that are all NaN
    signals = signals.dropna(axis=1, how='all')

    print("Columns after dropping all-NaN columns:")
    print(signals.columns)

    # Add the index as a column
    signals['index'] = signals.index

    if signals['mouse_id'].isin(['019', '020']).any():
        melted_session_signals = signals.melt(
            id_vars=['index', 'mouse_id', 'session_date', 'change_sizes_TF', 'change_category'], 
            var_name='roi', value_vars=['zscored_G0_clean_signal_dff', 'zscored_G2_clean_signal_dff', 'zscored_G4_clean_signal_dff', 'zscored_G5_clean_signal_dff']
        )
    else:
        melted_session_signals = signals.melt(
            id_vars=['index', 'mouse_id', 'session_date', 'change_sizes_TF', 'change_category'], 
            var_name='roi', value_vars=['zscored_G0_clean_signal_dff', 'zscored_G2_clean_signal_dff']
        )
    
    melted_session_signals.columns = ['seconds from event', 'mouse id', 'session date', 'change size', 'change category', 'hemisphere', 'zscored signal']
    
    # Use numpy.where to set the 'region' column based on the 'hemisphere' column
    melted_session_signals['region'] = np.where(
        melted_session_signals['hemisphere'].isin(['zscored_G0_clean_signal_dff', 'zscored_G2_clean_signal_dff']), 
        'DMS', 
        'VLS'
    )
    melted_session_signals['outcome'] = behave_event
    return melted_session_signals

def plot_melted_session_signals(melted_session_signals, behave_event):
    sns.set_context('talk')
    palette = sns.color_palette('bone_r', n_colors=5)  # Generate a palette with 10 colors
    colors_sizes = ["#808080","#ffa500","#ff8c00","#ff6347","#e60000","#990000"]
    colors_categories = ["#808080","#ffa500","#e60000"]

    mouse_id = melted_session_signals['mouse id'].iloc[0]

    plt.figure(figsize=(1.81, 1.83))
    if behave_event == 'hit' or behave_event == 'miss' or behave_event == 'change':
        g = sns.relplot(x='seconds from event', y='zscored signal', data=melted_session_signals, row='hemisphere', kind='line' ,hue='change category',palette=colors_categories, hue_order=['no_change', 'small', 'big'])
        plt.legend(loc='upper right', bbox_to_anchor=(2.0, 1), frameon=False, fontsize='x-small')
        
    else:
        g = sns.relplot(x='seconds from event', y='zscored signal', data=melted_session_signals, row='hemisphere', kind='line')
    
    # Iterate through each axis to set individual y-limits and add vertical lines
    for ax in g.axes.flatten():
        # # ax.set_ylim(-1, 2.5)  # Automatically set y-limits based on data range for each subplot
        # ax.set_ylim(0.3, 0.8)
        ax.axvline(x=0.00, color='black', linestyle='--')  # Add vertical line to each subplot
        sns.despine(ax=ax)  # Apply despine to each axis
        title = ax.get_title()
        ax.set_title(title, pad=20)

    
    plt.suptitle(f'mean signal around {behave_event} for {mouse_id}')
    plt.tight_layout(pad=0.5)
    plt.show()