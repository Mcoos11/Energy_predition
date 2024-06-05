from datetime import datetime
import pandas as pd
import numpy as np
import os
import multiprocessing as mp

DATA_DIR_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'data')

def prapare_datasets(input_data, df_headers, return_dict, procnum):
    print(f'proc {procnum} - start')
    features = ["dayofyear", "month", "weekofyear", "quarter", "year", "day", "weekday", "hour", "minute"]
    input_data = input_data.drop_duplicates(subset="tstp")# make unique
    input_data[features] = pd.to_datetime(input_data[df_headers[1]]).apply(
        lambda row: pd.Series({
            "dayofyear":row.dayofyear, 
            "month":row.month, 
            "weekofyear":row.weekofyear, 
            "quarter":row.quarter, "year":row.year, 
            "day":row.day, "weekday":row.weekday(), 
            "hour":row.hour, 
            "minute":row.minute 
            }))
    
    input_data = input_data.drop(['tstp'], axis=1)
    return_dict[procnum] = input_data

def save_dataset(datasets, data):
    for dataset in datasets:
        data.loc[data["LCLid"] == dataset].to_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets', f'{dataset}.csv'), sep=';', encoding="utf-8", mode="a", index=False)

if __name__ == '__main__':           
    mp.freeze_support()
    cpus = mp.cpu_count()
    files_names = [f for f in os.listdir(DATA_DIR_PATH) if os.path.isfile(os.path.join(DATA_DIR_PATH, f))]
    for file in files_names[:1]:
            input_data = pd.read_csv(os.path.join(DATA_DIR_PATH, file))
            df_headers = pd.read_csv(os.path.join(DATA_DIR_PATH, file)).columns.tolist()
            
            procs = list()
            manager = mp.Manager()
            return_dict = manager.dict()
            for count, part in enumerate(np.array_split(input_data, cpus)):
                procs.append(mp.Process(target=prapare_datasets, args=(part, df_headers, return_dict, count+1, )))
                procs[count].start()
                
            for proc in procs:
                proc.join()
                
            input_data = pd.concat(return_dict.values())
            print(f'{file} - done')
            
            datasets_filenames = input_data["LCLid"].unique()
            procs = list()
            for count, datasets in enumerate(np.array_split(datasets_filenames, cpus)):
                procs.append(mp.Process(target=save_dataset, args=(datasets, input_data, )))
                procs[count].start()
                
            for proc in procs:
                proc.join()
                
            print(f'{file} files - save')