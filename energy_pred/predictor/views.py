from django.shortcuts import render, redirect
from django import db, setup
from django.core.paginator import Paginator
setup()
from .models import EnergySpending
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from .prediction import time_prediction
from datetime import datetime, timedelta

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main_menu(request):
    hauseholds = EnergySpending.objects.order_by().values('hausehold').distinct()
    hause_energy = dict()
        
    paginator = Paginator(hauseholds, 10)
    page = request.GET.get('page')
    hauseholds = paginator.get_page(page)
    print(hauseholds)
    
    for hausehold in hauseholds:
        hausehold = hausehold['hausehold']
        energy_spend = 0.0
        for energy in EnergySpending.objects.filter(hausehold=hausehold):
            energy_spend += energy.energy
        hause_energy[str(hausehold)] = energy_spend
        
    return render(request, "predictor/main_menu.html", {'object_list': hauseholds, 'data': hause_energy})

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

def create_model(hauseholds):
    for hausehold in hauseholds:
        file_path = f'{DIR_PATH}\datasets\{hausehold}.csv'
        input_data = pd.read_csv(file_path, header=0, sep=';')
        restored_data = restore_data(input_data)
        
        for index, row in restored_data.iterrows():
            obj = EnergySpending(
                hausehold = row["LCLid"],
                energy = float(row['energy(kWh/hh)']),
                time = pd.to_datetime(f'{row["day"]}-{row["month"]}-{row["year"]} {row["hour"]}:{row["minute"]}', format='%d-%m-%Y %H:%M'),
            )
            obj.save()

def create_database(request):
    hauseholds = [f[:-4] for f in os.listdir(f'{DIR_PATH}\datasets') if os.path.isfile(os.path.join(f'{DIR_PATH}\datasets', f))]
    cpus = mp.cpu_count() 
    procs = list()
    db.connections.close_all()
    for count, part in enumerate(np.array_split(hauseholds, cpus)):
        procs.append(mp.Process(target=create_model, args=(part, )))
        procs[count].start()
                
    for proc in procs:
        proc.join()
            
    return redirect("predictor:main-menu")

def next_days_predict(date_list, hause, next_days):
    for date in date_list:
        next_days[date.strftime("%d-%m-%Y %H:%M")] = time_prediction(date, hause)
    return next_days
    
def details(request, hause):
    today = datetime.strptime("04.06.2012 00:00", "%d.%m.%Y %H:%M")
    last_days = dict()
    objects = EnergySpending.objects.filter(time__range=(today - timedelta(days=2), today))
    for obj in objects:
        last_days[obj.time.strftime("%d-%m-%Y %H:%M")] = obj.energy
    
    datetime_list = []
    delta = timedelta(hours=1)
    today = start_date = datetime.today()
    while (start_date <= (today + timedelta(days=2))):
        datetime_list.append(start_date)
        start_date += delta

    manager = mp.Manager()
    return_dict = manager.dict()
    cpus = mp.cpu_count() 
    procs = list()
    for count, part in enumerate(np.array_split(datetime_list, cpus)):
        procs.append(mp.Process(target=next_days_predict, args=(part, hause, return_dict)))
        procs[count].start()
                
    for proc in procs:
        proc.join()
    
    return render(request, "predictor/details.html", {"hause": hause, "last_days": last_days, "next_days": return_dict})