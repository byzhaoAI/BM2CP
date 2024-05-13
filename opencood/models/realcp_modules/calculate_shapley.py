#########################################################
### MULTI-TOUCH ATTRIBUTION MODEL USING SHAPLEY VALUE ###
#########################################################
# %%
### import packages
import itertools
from pathlib import Path
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#### setting up the formulas for shapley value
#######################################################################################################################

def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in itertools.combinations(List, i+1)]
    return PS
  
#######################################################################################################################

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
    
#######################################################################################################################

def v_function(A,C_values):
    '''
    This function computes the worth of each coalition.
    inputs:
            - A : a coalition of channels.
            - C_values : A dictionnary containing the number of conversions that each subset of channels has yielded.
    '''
    subsets_of_A = subsets(A)
    worth_of_A=0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    return worth_of_A
      
#######################################################################################################################

def subsets(s):
    '''
    This function returns all the possible subsets of a set of channels.
    input :
            - s: a set of channels.
    '''
    if len(s)==1:
        return s
    else:
        sub_channels=[]
        for i in range(1,len(s)+1):
            sub_channels.extend(map(list,itertools.combinations(s, i)))
    return list(map(",".join,map(sorted,sub_channels)))
  
#######################################################################################################################

def calculate_shapley(df, channel_name, conv_name):
    '''
    This function returns the shapley values
            - df: A dataframe with the two columns: ['channel_subset', 'count'].
            The channel_subset column is the channel(s) associated with the conversion and the count is the sum of the conversions. 
            - channel_name: A string that is the name of the channel column 
            - conv_name: A string that is the name of the column with conversions
            **Make sure that that each value in channel_subset is in alphabetical order. Email,PPC and PPC,Email are the same 
            in regards to this analysis and should be combined under Email,PPC.
            
    '''
    # casting the subset into dict, and getting the unique channels
    c_values = df.set_index(channel_name).to_dict()[conv_name]
    channels = df[channel_name].apply(lambda x: x.split(",")).explode().unique()

    print(c_values)
    print(channels)
    print(power_set(channels), len(power_set(channels)))
    
    v_values = {}
    for A in power_set(channels): #generate all possible channel combination
        v_values[','.join(sorted(A))] = v_function(A, c_values)
    
    print(v_values)

    n = len(channels) #no. of channels
    shapley_values = defaultdict(int)

    for channel in channels:
        for A in v_values.keys():
            if channel not in A.split(","):
                cardinal_A=len(A.split(","))
                A_with_channel = A.split(",")
                A_with_channel.append(channel)            
                A_with_channel=",".join(sorted(A_with_channel))
                print(A, cardinal_A, A_with_channel)

                weight = (factorial(cardinal_A)*factorial(n-cardinal_A-1)/factorial(n)) # Weight = |S|!(n-|S|-1)!/n!
                contrib = (v_values[A_with_channel]-v_values[A]) # Marginal contribution = v(S U {i})-v(S)
                shapley_values[channel] += weight * contrib
        # Add the term corresponding to the empty set
        shapley_values[channel]+= v_values[channel]/n 


    
        
    return shapley_values


def generate_subsets(lst):  
    # 计算子集的总数（2的列表长度次方减去1，因为不包括空集）  
    num_subsets = 2 ** len(lst) - 1
    # 使用位运算生成所有子集  
    subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
    return subsets


if __name__ == '__main__':
    ### reading in dataset
    data_path = 'datasets_marketing.csv'
    data_path_resolve = Path(data_path).resolve()
    data_raw = pd.read_csv(data_path_resolve)

    # %%
    ###  extracting the needed field
    columns = ['user_id', 'date_served', 'marketing_channel', 'converted']
    data = data_raw[columns].copy()
    print(f"Data dimension: {data.shape}")

    ### dropping null values
    data.dropna(axis=0, inplace=True)
    print(f"Data dimension after dropping null values: {data.shape}")

    ### relabel conversion to 1/0
    data['converted'] = data['converted'].astype('int') 
    ### converting date_served into date format
    data['date_served'] = pd.to_datetime(data['date_served'], format='%m/%d/%y', errors='coerce')

    # %%
    ### create a channel mix conversion table 
    # first level - sort
    data_lvl1 = data[['user_id', 'marketing_channel', 'converted']].sort_values(by=['user_id', 'marketing_channel'])
    # second level - groupby userid, concat distinct marketing channel and label if any conversion took place with this channel mix
    data_lvl2 = data_lvl1.groupby(['user_id'], as_index=False).agg({'marketing_channel': lambda x: ','.join(map(str,x.unique())),
                                                                    'converted':max})
    data_lvl2.rename(columns={'marketing_channel':'marketing_channel_subset'}, inplace=True)
    # third level - summing up the conversion which took place for each channel mix
    data_lvl3 = data_lvl2.groupby(['marketing_channel_subset'], as_index=False).agg(sum)

    print(data_lvl3)
    ### calculating the shapley value of the channel
    shapley_dict = calculate_shapley(data_lvl3, 'marketing_channel_subset', 'converted')
    shapley_result = pd.DataFrame(list(shapley_dict.items()), columns=['channel', 'shapley_value'])


    ### visualizing the results
    # sns.set_style("white")
    plt.subplots(figsize=(15,8))
    s = sns.barplot(x='channel', y='shapley_value', data=shapley_result)

    for idx, row in shapley_result.iterrows():
        s.text(row.name, row.shapley_value +5, round(row.shapley_value,1), ha='center', color='darkslategray', fontweight='semibold')
    plt.title("ADVERTISING CHANNEL'S SHAPLEY VALUE", 
            fontdict={'fontfamily': 'san-serif', 'fontsize': 15, 'fontweight': 'semibold', 'color':'#444444'}, 
            loc='center', pad=10)
    plt.show()
