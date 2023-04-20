import helpers as hp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# path_to_parent_folder = 'X:/public/projects/BeJG_20230130_VisDetect/' (explicit path - '\\ceph-gw02.hpc.swc.ucl.ac.uk\mrsic_flogel\public\projects\BeJG_20230130_VisDetect\' )
# path_to_parent_folder = 'Y:\public\projects\MiLo_20211201_DMDM_CausalCortex/All_Mouse_Behaviour_Data_Raw/Main/'

path_to_parent_folder = str(input('enter path to parent folder (including final slash):'))
return_dict_or_df = str(input('return dict, df or all? (dict, df or all)'))

if return_dict_or_df != 'dict':
    choose_outcome = str(input('choose outcome of trials to include in dataframe (all, Hit,Miss,FA,Abort,Ref )'))
else: choose_outcome = 'all'

behave_data = hp.get_behave_data(path_to_parent_folder = path_to_parent_folder, choose_outcome = choose_outcome, return_dict_or_df = return_dict_or_df) #,include_auto_reward_trials = True)

# behave_data

sns.set_style('ticks')
# sns.color_palette("Set2")
# sns.displot(data = behave_data[(behave_data['session_trial_outcome']!='Miss') & (behave_data['session_trial_outcome']!='Ref')], x = 'reaction_time', hue = 'change_size_TF',col='session_trial_outcome', palette = sns.color_palette('rocket'), kind='hist',binwidth = 0.05, alpha = 0.3 ,kde = True, stat = 'percent',common_norm=False)
sns.displot(data = behave_data[behave_data['session_trial_outcome']=='Hit'], x = 'reaction_time', hue = 'change_size_TF', palette = sns.color_palette('rocket'), kind='hist',binwidth = 0.05, alpha = 0.3 ,kde = True, stat = 'percent',common_norm=False,col = 'change_size_TF',row='mouse_id')

sns.show()  # required now not in notebook
