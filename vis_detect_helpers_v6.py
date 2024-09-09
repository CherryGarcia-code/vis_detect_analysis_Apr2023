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

def find_folder(mouse):
    """Recursively search for a folder in the given drive and find the drive where the folder is located."""
    system_drive = os.getenv("SystemDrive") + '\\' # Get the drive letter of the local disk (system drive) on Windows.
    for root, dirs, files in os.walk(system_drive):
        if mouse in dirs:
            mouse_dir = os.path.join(root, mouse) + '\\'
            cohort = os.path.basename(root)
            return mouse_dir, cohort

def get_session_files(mouse, cohort_dir=False, timestamp=False,):
    if cohort_dir:
        mouse_dir = os.path.join(cohort_dir, mouse) + '\\'
    else:
        mouse_dir, _ = find_folder(mouse)
    # Glob all session settings and trials files in the directory
    session_settings_files = glob(os.path.join(mouse_dir, '*__session_settings.json'))
    trials_files = glob(os.path.join(mouse_dir, '*__trials.json'))

    # Filter files by timestamp if a specific timestamp is provided
    if timestamp:
        # Assuming the timestamp is part of the file's name, adjust the condition to match your file naming convention
        session_settings_files = [file for file in session_settings_files if timestamp in os.path.basename(file)]
        trials_files = [file for file in trials_files if timestamp in os.path.basename(file)]

    print('number of files: session_settings: ', len(session_settings_files), 'trials: ', len(trials_files))
    
    return session_settings_files, trials_files

def parse_timestamp(date_str, time_str):
    # Combine date and time strings into a single datetime string
    datetime_str = f"{date_str}{time_str}"
    # Specify the format corresponding to 'YYYYMMDDHHMMSS'
    datetime_format = "%Y%m%d%H%M%S"
    # Parse the datetime string into a datetime object
    return datetime.strptime(datetime_str, datetime_format)

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
        return iti_values, reaction_times, reaction_times_from_reference_start, outcomes, change_times, change_sizes_TF, TF_vectors



def process_session_data(session_settings_files, trials_files, cohort_dir=False):
    session_data_df = pd.DataFrame()
    for i in range(len(trials_files)):
        # Example for one session
        print('******************   NEW   SESSION   ******************')

        session_settings = load_json_data(session_settings_files[i])
        trials_data = load_json_data(trials_files[i])
    
        # Extract the base file name without the directory
        base_filename = os.path.basename(session_settings_files[i])

        
        mouse_id, session_date,  session_time,  session_timestamp, auto_rewd, punishment, protocol = extractSessionMetadata(session_settings, base_filename)
        iti_values, reaction_times, reaction_times_from_reference_start, outcomes, change_times, change_sizes_TF, TF_vectors = extractTrialsData(trials_data)
        if cohort_dir:
            cohort = os.path.basename(cohort_dir)
        else:
            _, cohort = find_folder('BG_' + mouse_id)

        
        print('mouse_id is: ',mouse_id)
        print('mouse cohort is: ', cohort)
        print('session_date is: ', session_date)
        print('session_time is: ', session_time)
        print('session timestamp is: ',session_timestamp)
        print('len iti_values:', len(iti_values))
        print('auto rewd is: ',auto_rewd)
        print('punishment is: ',punishment)
        print('protocol is: ',protocol)
        print('unique outcomes are: ',np.unique(outcomes))
    

        processed_data = {
            'mouse_id': mouse_id,
            'cohort': cohort,
            'session_date': session_date,
            'session_time': session_time,
            'session_timestamp': session_timestamp,
            'auto_rewd': auto_rewd,
            'punishment': punishment,
            'protocol': protocol,
            'iti_values': iti_values,
            'change_times': change_times,
            'change_sizes_TF': change_sizes_TF,
            'outcomes': outcomes,
            'reaction_times': reaction_times,
            'reaction_times_from_reference_start': reaction_times_from_reference_start,
            'TF_vectors': TF_vectors                
        }

        session_data_df = pd.concat([session_data_df, pd.DataFrame(processed_data)])
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
               output_dir = os.getcwd() + '/photom_plots'):
    
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


# Function to extract a window of data around an event timestamp and pad it to a fixed size
def extract_signal_window_from_trial_df(df, event_timestamp, window_size=2.0, fixed_window_size=401):
    # Get the nested DataFrame for the trial
    trial_data = df['dff_data']
    # Calculate the window start and end times
    window_start = event_timestamp - window_size
    window_end = event_timestamp + window_size
    # Ensure the window start is after the baseline timestamp
    # window_end= max(window_start, df.loc[trial_idx, 'baseline_on_timestamps'])
    # Ensure the window end is at least a second before the reaction timestamp:
    # window_end = min(window_end, df['baseline_on_timestamps']-1)
    # Filter the trial data to the window
    window_data = trial_data[(trial_data['SystemTimestamp'] >= window_start) &
                             (trial_data['SystemTimestamp'] <= window_end)]
    window_data['mouse_id'] = df['mouse_id']
    window_data['session_date'] = df['session_date']
    window_data['cell_type'] = df['cell_type']
    window_data['change_sizes_TF'] = df['change_sizes_TF']
    # Pad the window data to the fixed size
    padding_size = fixed_window_size - len(window_data)
    if padding_size > 0:
        padding_index = np.linspace(-window_size, window_size, padding_size, endpoint=True)
        padding_df = pd.DataFrame(index=padding_index, columns=window_data.columns)
        window_data = pd.concat([window_data, padding_df]).sort_index().fillna(method='ffill').fillna(method='bfill').tail(fixed_window_size)
    window_data = window_data.set_index(np.linspace(-window_size, window_size, fixed_window_size, endpoint=True))
    
    return window_data