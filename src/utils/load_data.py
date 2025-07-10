import pandas as pd   
import os     
import random as random  
def load_data_ran(data_path, num_files=5, random_start=0, range_end=1684): #random_start is the starting number of the files to be loaded, range_end is the ending number of the files to be loaded
    available_range = range(random_start, range_end)
    random_numbers = random.sample(available_range, num_files)
    dataframes = []
    for i in random_numbers:
        file_path =os.path.join(data_path, f'NamedlmfdbDataRanks{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=';', low_memory=False)
            
            dataframes.append(df)
    result = pd.concat(dataframes, axis=0)
    
    return result
#load_data_ran is used to load the data from the random files
def load_data(data_path, range_start, range_end): #range_start is the starting number of the files to be loaded, range_end is the ending number of the files to be loaded
    data_frames = []
    for i in range(range_start, range_end):
        file_path = os.path.join(data_path, f'NamedlmfdbDataRanks{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=';', low_memory=False)
            #df.columns = (
                #df.columns
                #.str.strip()
                #.str.replace(r"\s+", " ", regex=True) 
            #)
            #obj_cols = df.select_dtypes(include='object').columns
            #df[obj_cols] = (
                #df[obj_cols]
                #.apply(lambda col: col.str.strip()
                                #.str.replace(r'\s+', ' ', regex=True))
            #)
            data_frames.append(df)
    result = pd.concat(data_frames, axis=0)
    return result
#create_data_by_feature_target is used to create the data by the feature and target
def create_data_by_feature_target(data, feature, target):
    data_by_feature_target = data[feature + target]
    data_by_feature_target = data_by_feature_target.dropna()
    return data_by_feature_target
#combine_data_by_feature_target is used to combine the data by the feature and target
def combine_data_by_feature_target(feature, target,data_path,range_start,range_end):
    data_combine = pd.DataFrame()
    for i in range(range_start, range_end):
        file_path = os.path.join(data_path, f'NamedlmfdbDataRanks{i}.csv')
        if os.path.exists(file_path):
            # Read CSV, ensuring spaces after delimiters are handled
            df = pd.read_csv(file_path, sep=';', skipinitialspace=True, low_memory=False)
            df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
            #df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col, axis=0)
            for c in df.select_dtypes(include="object").columns:
                df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Clean column names to match the expected format
            #df.columns = (
                #df.columns
                  #.str.strip()  # Remove leading/trailing spaces
                  #.str.replace(r"\s+", " ", regex=True)  # Collapse multiple internal spaces
            #)
            
            data_by_feature_target = create_data_by_feature_target(df, feature, target)
            data_combine = pd.concat([data_combine, data_by_feature_target], axis=0)

            print(f"Loaded file {i} with shape {df.shape}")
            print(f"Combined data shape: {data_combine.shape}")
    
    return data_combine
