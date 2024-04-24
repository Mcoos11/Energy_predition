from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib, os
import numpy as np
import multiprocessing as mp

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# prediction fun
def RFR(X, y=None, city=None):
    # models
    file_path = os.path.join(DIR_PATH, "models", f'RFR_model_{city}.sav')
    loaded_model = joblib.load(file_path)
    # predictions
    RFR_predictions = loaded_model.predict(X)

    if y is not None:
        return RFR_predictions, mean_squared_error(y, RFR_predictions, squared=False)
    else:
        return RFR_predictions

def KNN(X, y=None, city=None):
    # models
    file_path = os.path.join(DIR_PATH, "models", f'KNN_model_{city}.sav')
    loaded_model = joblib.load(file_path)
    # data to models and preprocessing
    file_path = os.path.join(DIR_PATH, "models", "scalers", f'KNN_scaler_{city}.sav')
    scaler = joblib.load(file_path)
    X = scaler.transform(X)
    # predictions
    KNN_predictions = loaded_model.predict(X)

    if y is not None:
        return KNN_predictions, mean_squared_error(y, KNN_predictions, squared=False)
    else:
        KNN_predictions

def main(cities):
    for city in cities:
        # results
        df_out = pd.DataFrame()
        df_tmp = pd.DataFrame()
        predict_value = "energy(kWh/hh)"
        
        X_path = os.path.join(DIR_PATH, "test_data", f'X_{city}_test.csv')        
        X_test = pd.read_csv(X_path, header=0, sep=',')
        y_path = os.path.join(DIR_PATH, "test_data", f'y_{city}_test.csv')
        y_test = pd.read_csv(y_path, header=0, sep=',')
        
        to_one_col = lambda df_date: [f'{df_row["year"]}-{df_row["month"]}-{df_row["day"]} {df_row["hour"]}:{df_row["minute"]}' for _, df_row in df_date.iterrows()]

        RFR_predictions, RFR_rmse = RFR(X_test, y_test, city)
        print(f'RFR {city} rmse:', RFR_rmse)


        KNN_predictions, KNN_rmse = KNN(X_test, y_test,city)
        print(f'KNN {city} rmse:', KNN_rmse)
        
        results = [to_one_col(X_test), y_test[predict_value].tolist(), RFR_predictions, KNN_predictions]
        names  = ['timestamp', f'original_value', f'RFR_pred_value', f'KNN_predictions']
        for index, (name, data) in enumerate(zip(names, results)):
            df_tmp.insert(index, name, data, True)

        df_out = pd.concat([df_out, df_tmp], axis=1)
            
        df_out.to_csv(f'after-predictions/{city}-after-prediction.csv', sep=';', index=False, encoding='utf-8')
        
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