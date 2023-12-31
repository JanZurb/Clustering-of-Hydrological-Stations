import pandas as pd
import glob
import scipy
import matplotlib as plt
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import RobustScaler




def importfiles(path):
    all_files = glob.glob(path)
    list_allfiles = []
    for files in all_files:
        df = pd.read_csv(files, sep=';', skiprows=8, encoding = "latin1")
        list_allfiles.append(df)
    all_stations_df = pd.concat(list_allfiles, axis=0, ignore_index=True)
    return all_stations_df



def create_flow_temp_df(path_flow, path_temp):

    all_temp_stations_df = importfiles(path_temp)
    all_flow_stations_df = importfiles(path_flow)

    all_temp_stations_df.rename(columns={'Wert': 'Flow_Wert'},inplace=True)

    unnamed_cols  =  all_temp_stations_df.columns.str.contains('Unnamed')
    all_temp_stations_df.drop(all_temp_stations_df[all_temp_stations_df.columns[unnamed_cols]], axis=1, inplace=True)
    all_temp_stations_df.drop(columns= ['Gewässer', 'Stationsname'], axis=1, inplace=True)

    all_flow_stations_df.rename(columns={'Wert': 'Temp_Wert'},inplace=True)

    flow_temp_data = pd.merge(all_flow_stations_df, all_temp_stations_df, on=['Zeitstempel', 'Stationsnummer'])
    flow_temp_data.rename(columns={'Gewässer_x': 'Gewässer'}, inplace=True)

    flow_temp_data = flow_temp_data[['Zeitstempel', 'Stationsname', 'Stationsnummer', 'Flow_Wert', 'Temp_Wert', 'Gewässer']]
    return flow_temp_data



def get_station_df (station_number, flow_temp_data):
    station_df = flow_temp_data[flow_temp_data['Stationsnummer'] == station_number]
    station_df = station_df[['Stationsnummer', 'Zeitstempel', 'Flow_Wert', 'Temp_Wert']]
    return station_df


def get_time_period(station_number, start_date, end_date, flow_temp_data):
    station_df = get_station_df(station_number, flow_temp_data)
    station_df['Zeitstempel'] = pd.to_datetime(station_df['Zeitstempel'])
    station_df = station_df.set_index('Zeitstempel')
    station_df = station_df.loc[start_date:end_date]
    return station_df

def get_daily_averaged_df(station_number, flow_temp_data):
    station_data = get_station_df(station_number, flow_temp_data)
    station_data['Zeitstempel'] = pd.to_datetime(station_data.Zeitstempel)
    #group by day and calculate the mean per day of the year
    station_data_average_year = station_data.groupby(station_data.Zeitstempel.dt.dayofyear).mean(numeric_only=True)
    return station_data_average_year


def get_running_mean_df(station_number, window, flow_temp_data, Wert):
    daily_averaged_data = get_daily_averaged_df(station_number, flow_temp_data)
    #add the last 15 days of the year to the beginning of the year
    expanded_data = daily_averaged_data.iloc[-window+1:].append(daily_averaged_data)  
    daily_averaged_data[Wert] = daily_averaged_data[Wert].rolling(window=window).mean()
    expanded_data = expanded_data.rolling(window=window).mean().dropna().reset_index(drop=True)
    return expanded_data


# def normalize_data(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         #exclude feature_names that include 'day' or 'time' string
#         if feature_name.find('day') != -1 or feature_name.find('Stationsnummer')!= -1:
#             continue
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         mean_value = df[feature_name].mean()
#         std_value = df[feature_name].std()
#         result[feature_name] = (df[feature_name] - mean_value) / (std_value)
#     return result


# version with robust scaler
def normalize_data(df):
    #remove Stationsnummer column
    stationsnummer = df['Stationsnummer']
    day_df = pd.DataFrame()
    df = df.drop(['Stationsnummer'], axis=1)
    #remove the day columns as they are already normalized
    for feature_name in df.columns:
        #exclude feature_names that include 'day' or 'time' string
        if feature_name.find('day') != -1:
            #create df with all the day columns
            day_df[feature_name] = df[feature_name]
            df.drop([feature_name], axis=1, inplace=True)
            continue
    #use robust scalar ro normalize the data
    robust_scaler = RobustScaler(quantile_range=(5,95))
    robust_scaler.fit(df)
    normalized_data = robust_scaler.transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=df.columns)
    #add the day columns again
    normalized_data = pd.concat([normalized_data, day_df], axis=1)
    normalized_data['Stationsnummer'] = stationsnummer
    return normalized_data

def robust_normalize_data(df):
    #remove Stationsnummer column
    stationsnummer = df['Stationsnummer']
    df = df.drop(['Stationsnummer'], axis=1)
    #make use of the robust scaler from sklearn
    robust_scaler = RobustScaler()
    robust_scaler.fit(df)
    normalized_data = robust_scaler.transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=df.columns)

    normalized_data['Stationsnummer'] = stationsnummer
    return normalized_data

def force_normalize_data(df):
    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / (std_value)
    return result



def create_mds_represntation(normalized_data):
    stationsnummer = normalized_data['Stationsnummer']
    normalized_data = normalized_data.drop(['Stationsnummer'], axis=1)
    X = normalized_data.values

    mds = MDS(random_state=0)
    X_transform = mds.fit_transform(X)

    #add mds coordinates to new df
    mds_representation = pd.DataFrame(columns=['x_MDS', 'y_MDS'])
    mds_representation['x_MDS'] = X_transform[:,0]
    mds_representation['y_MDS'] = X_transform[:,1]
    mds_representation['Stationsnummer'] = stationsnummer
    return mds_representation

# get number of peaks with scipy
def get_number_of_peaks(data, Wert, distance):
    num_peaks = scipy.signal.find_peaks(data[Wert],distance = distance)[0].size
    return num_peaks

# calculate similarity of each year to the mean year
def get_similarity_to_mean_year(original_station_data, Wert):
    
    #calculate mean year
    mean_year = get_daily_averaged_df(original_station_data['Stationsnummer'].iloc[0], original_station_data)
    original_station_data['Zeitstempel'] = pd.to_datetime(original_station_data.Zeitstempel)
    #calculate similarity of each year to the mean year
    sum_of_differences = 0
    for year, group in original_station_data.groupby(original_station_data.Zeitstempel.dt.year):
        difference = np.sum(np.abs(group[Wert].values - mean_year[Wert].values[:len(group[Wert].values)])) 
        difference = difference/mean_year[Wert].values[:len(group[Wert].values)].mean()
        sum_of_differences = sum_of_differences+difference
    return sum_of_differences

def get_max_slope(station_data, Wert, days):
    #calculate difference between following days of the Wert
    slope = []
    for i in range(len(station_data)-days-2):
        slope.append(station_data[Wert][i+2]-station_data[Wert][i+2+days])#sketchy thingy
    
    array = np.array(slope)
    max_slope = np.nanmax(array)

    return max_slope

def get_min_slope(station_data, Wert, days):
    #calculate difference between following days of the Wert
    slope = []
    for i in range(len(station_data)-days-2):
        slope.append(station_data[Wert][i+2]-station_data[Wert][i+2+days])#sketchy thingy
    
    array = np.array(slope)
    min_slope = np.nanmin(array)

    return min_slope

def get_day_of_max(station_df, Wert):
    #get index of the day with the highest value
    day_of_max = station_df[Wert].idxmax()
    return day_of_max

def get_day_of_min(station_df, Wert):
    #get index of the day with the highest value
    day_of_min = station_df[Wert].idxmin()
    return day_of_min
    





def get_time_above_average(station_df, Wert):
    #calculate average
    average = np.mean(station_df[Wert])
    #calculate border
    border = average 

    over_limit = station_df[station_df[Wert] > border]
    days_above_average = len(over_limit)

    return days_above_average 

def get_time_first_upward_crossing_mean(stations_df, Wert):
    #calculate average
    average = np.mean(stations_df[Wert])
    #get first upward crossing after minimum
    day_of_minimum = get_day_of_min(stations_df, Wert)
    #slice df from day of minimum onwards
    before_minimum = stations_df.iloc[:day_of_minimum]
    after_minimum = stations_df.iloc[day_of_minimum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_minimum, before_minimum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] > average]
    #get the original index of the first entry greater than the average
    first_upward_crossing_after_min = stations_df_greater_than_average.index[0]
    
    return first_upward_crossing_after_min



# def get_time_first_upward_crossing_mean(stations_df, Wert):
#     #calculate average
#     average = np.mean(stations_df[Wert])
    
#     #get first upward crossing of average
#     #case 1: first value is already greater than average
#     if stations_df[Wert][0] < average:
#         first_upward_crossing = stations_df[stations_df[Wert] > average].index[0]

#     #case 2: first value is greater than average
#     else:
#         first_upward_crossing = stations_df[stations_df[Wert] < average].index[-1]
#     return first_upward_crossing

def get_time_first_downward_crossing_mean(stations_df, Wert):
    #calculate average
    average = np.mean(stations_df[Wert])
    day_of_maximum = get_day_of_max(stations_df, Wert)

    before_maximum = stations_df.iloc[:day_of_maximum]
    after_maximum = stations_df.iloc[day_of_maximum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_maximum, before_maximum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] < average]
    #get the original index of the first entry greater than the average
    first_downward_crossing_after_max = stations_df_greater_than_average.index[0]
    
    return first_downward_crossing_after_max
    
# def get_time_first_downward_crossing_mean(stations_df, Wert):
#     #calculate average
#     average = np.mean(stations_df[Wert])
#     #get first downward crossing of average
#     #case 1: first value is already less than average
#     if stations_df[Wert][0] > average:
#         first_downward_crossing = stations_df[stations_df[Wert] < average].index[0]

#     #case 2: first value is greater than average
#     else:
#         first_downward_crossing = stations_df[stations_df[Wert] > average].index[-1]  
#     return first_downward_crossing

def get_time_first_upward_crossing_highquantile(stations_df, Wert):
    
    #get value in station_df that is greater than 75% quantile
    quantile = (np.max(stations_df[Wert]) - np.min(stations_df[Wert])) * 0.75 + np.min(stations_df[Wert])
    #get first upward crossing of average
    day_of_minimum = get_day_of_min(stations_df, Wert)
    #slice df from day of minimum onwards
    before_minimum = stations_df.iloc[:day_of_minimum]
    after_minimum = stations_df.iloc[day_of_minimum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_minimum, before_minimum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] > quantile]
    #get the original index of the first entry greater than the average
    first_upward_crossing_after_min = stations_df_greater_than_average.index[0]
    return first_upward_crossing_after_min

def get_time_first_downward_crossing_highquantile(stations_df, Wert):
    #calculate average
    average = np.mean(stations_df[Wert])
    quantile = (np.max(stations_df[Wert]) - np.min(stations_df[Wert])) * 0.75 + np.min(stations_df[Wert])

    
    day_of_maximum = get_day_of_max(stations_df, Wert)

    before_maximum = stations_df.iloc[:day_of_maximum]
    after_maximum = stations_df.iloc[day_of_maximum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_maximum, before_maximum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] < quantile]
    #get the original index of the first entry greater than the average
    first_downward_crossing_after_max = stations_df_greater_than_average.index[0]
    
    return first_downward_crossing_after_max




def get_time_first_upward_crossing_lowquantile(stations_df, Wert):
    quantile = (np.max(stations_df[Wert]) - np.min(stations_df[Wert])) * 0.25 + np.min(stations_df[Wert])
    #get first upward crossing of average
    day_of_minimum = get_day_of_min(stations_df, Wert)
    #slice df from day of minimum onwards
    before_minimum = stations_df.iloc[:day_of_minimum]
    after_minimum = stations_df.iloc[day_of_minimum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_minimum, before_minimum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] > quantile]
    #get the original index of the first entry greater than the average
    first_upward_crossing_after_min = stations_df_greater_than_average.index[0]
    return first_upward_crossing_after_min

def get_time_first_downward_crossing_lowquantile(stations_df, Wert):
    #calculate average
    average = np.mean(stations_df[Wert])
    quantile = (np.max(stations_df[Wert]) - np.min(stations_df[Wert])) * 0.25 + np.min(stations_df[Wert])

    
    day_of_maximum = get_day_of_max(stations_df, Wert)

    before_maximum = stations_df.iloc[:day_of_maximum]
    after_maximum = stations_df.iloc[day_of_maximum:]
    #reconnect the two dfs with after_minimum first
    stations_df_reordered = pd.concat([after_maximum, before_maximum])
    #reduce the df to only entries greater than the average
    stations_df_greater_than_average = stations_df_reordered[stations_df_reordered[Wert] < quantile]
    #get the original index of the first entry greater than the average
    first_downward_crossing_after_max = stations_df_greater_than_average.index[0]
    
    return first_downward_crossing_after_max

    

def get_sin_cos_rep(day_feature):
    sin = np.sin(day_feature/365 * 2* np.pi)
    cos = np.cos(day_feature/365*2*np.pi)

    return sin,cos

#score functions
def get_max_and_num_clust(scores):
    #reduce scores entries to 20 scores is a df with two columns
    scores = scores.iloc[2:19]
    max_score = max(scores['silhouette_score'])
    id = scores['silhouette_score'].idxmax()
    num_clust = scores.loc[id]['num_clusters']
    return max_score, num_clust

def get_min_and_num_clust(scores):
    scores = scores.iloc[2:19]
    min_score = min(scores['davies_bouldin_score'])
    id = scores['davies_bouldin_score'].idxmin()
    num_clust = scores.loc[id]['num_clusters']
    return min_score, num_clust

def get_best_score(scores, score_name):
    if score_name == 'silhouette':
        scores = scores.iloc[2:14]
        max_score = max(scores['silhouette_score'])
        id = scores['silhouette_score'].idxmax()
        num_clust = scores.loc[id]['num_clusters']
        return max_score, num_clust
        
    elif score_name == 'davies_bouldin':
        scores = scores.iloc[2:14]
        min_score = min(scores['davies_bouldin_score'])
        id = scores['davies_bouldin_score'].idxmin()
        num_clust = scores.loc[id]['num_clusters']
        return min_score, num_clust

    else:
        print('wrong score_name')
        return None, None
