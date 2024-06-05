import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# models fun
def RFR(X_train, y_train, city=None):
    # models
    RFR_model = RandomForestRegressor()
    # data to models and preprocessing
    RFR_model.fit(X_train, y_train.values.ravel())
    # model saving
    file_path = os.path.join(DIR_PATH, "models", f'RFR_model_{city}.sav')
    joblib.dump(RFR_model, file_path)

def KNN(X_train, y_train, city=None):
    # models
    KNN_model = KNeighborsRegressor(n_neighbors=4)
    # data to models and preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    file_path = os.path.join(DIR_PATH, "models", "scalers", f'KNN_scaler_{city}.sav')
    joblib.dump(scaler, file_path)
    X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    KNN_model.fit(X_train, y_train)
    # model saving
    file_path = os.path.join(DIR_PATH, "models", f'KNN_model_{city}.sav')
    joblib.dump(KNN_model, file_path)
    
# imput missing data
def restore_data(df: pd.DataFrame):
    df = df.replace('Null', np.nan)
    grouped = df.groupby(["month"])
    out = pd.DataFrame(columns=df.columns)
    for month in range(1, 13):
        group_imputing = SimpleImputer(missing_values=np.nan, strategy="mean")
        try:
            current_month = grouped.get_group(month)
        except KeyError:
            continue
        group_imputing.fit(current_month[["energy(kWh/hh)"]])
        try:
            current_month[["energy(kWh/hh)"]] = group_imputing.transform(current_month[["energy(kWh/hh)"]])
        except ValueError:
            current_month[["energy(kWh/hh)"]] = current_month[["energy(kWh/hh)"]].fillna(0)
        out = pd.concat([out, current_month], axis=0)

    return out

def main(cities):
    for city in cities:
        file_path = f'{DIR_PATH}\datasets\{city}.csv'
        headers = pd.read_csv(file_path, sep=';').columns.tolist()
        input_data = pd.read_csv(file_path, header=0, sep=';')

        features = ["dayofyear", "month", "weekofyear", "quarter", "year", "day", "weekday", "hour", "minute"]
        restored_data = restore_data(input_data)            
        y = restored_data[["energy(kWh/hh)"]]
        X = input_data[features]

        # training data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        except ValueError:
            continue

        # saving test data
        X_path = os.path.join(DIR_PATH, "test_data", f'X_{city}_test.csv')
        X_test.to_csv(X_path, sep=',', index=False, encoding='utf-8')
        y_path = os.path.join(DIR_PATH, "test_data", f'y_{city}_test.csv')
        y_test.to_csv(y_path, sep=',', index=False, encoding='utf-8')

        # creating models
        RFR(X_train, y_train, city)
        KNN(X_train, y_train, city)
    

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    mp.freeze_support()
    
    cities = [f[:-4] for f in os.listdir(f'{DIR_PATH}\datasets') if os.path.isfile(os.path.join(f'{DIR_PATH}\datasets', f))]
    cpus = mp.cpu_count() 
    procs = list() 
    for count, part in enumerate(np.array_split(cities, cpus)):
        procs.append(mp.Process(target=main, args=(part, )))
        procs[count].start()
                
    for proc in procs:
        proc.join()
        