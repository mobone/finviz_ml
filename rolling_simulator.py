import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import itertools
from random import shuffle
import numpy as np
import scipy.stats
from sklearn.preprocessing import Imputer
import threading
import queue
import matplotlib.pyplot as plt
import sqlite3
from sklearn.externals import joblib
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import pandas as pd
import configparser
from datetime import datetime, timedelta
import threading
import queue



def read_config():
    config = configparser.ConfigParser()
    config.read('settings.cfg')

    ip = config['CREDS']['ip']
    port = int(config['CREDS']['port'])
    authSource = config['CREDS']['authSource']
    user = config['CREDS']['username']
    password = config['CREDS']['Password']

    return (ip, port, authSource, user, password)

ip, port, authSource, user, password = read_config()

client = MongoClient(ip, port=port, authSource=authSource, username=user, password=password)
db = client['finance']
coll = db['finviz']
spy = db['options_2']


def p2f(x):
    return float(x.strip('%'))

def clean_data(df):

    for i in df.columns:
        try:
            #print(i)
            df[i] = (df[i].replace(r'[KMB]+$', '', regex=True).astype(float) * \
               df[i].str.extract(r'[\d\.]+([KMB]+)', expand=False)
                 .fillna(1)
                 .replace(['K','M','B'], [10**3, 10**6, 10**9]).astype(int))
        except:
            pass

    for i in df.columns:
        try:
            df[i] = pd.to_numeric(df[i])
        except:
            pass

    for column in ['_id','_rev','Ticker','Root.1', '52W Range', 'Earnings','Sector','Industry','Optionable','Index','Ticker','Shortable']:
        if column in df.columns:
            df = df.drop(column, axis=1)

    return df

df = pd.read_csv('rolling_data.csv', converters={'col':p2f})
df = clean_data(df)
df = df[df.columns[1:]]
cols = list(df.columns)
cols.insert(0, cols.pop(cols.index('Root')))
df = df[cols]


imp = Imputer(missing_values='NaN', strategy='mean')
imp.fit(df[df.columns[1:]])
data = pd.DataFrame(imp.transform(df[df.columns[1:]]),columns=df.columns[1:])
df[df.columns[1:]] = data

conn = sqlite3.connect('results.db')
sql = 'select * from model_results order by mean desc'
models = pd.read_sql(sql,conn)

for key,model in models.iterrows():

    features = model['Features']
    perc_change = round(model['mean'],4)
    columns = model['columns']
    model_name = model['Name']
    #features_str = str(features)[2:-2].replace('/','-o-').replace("', '",'_')
    file_name = 'models/'+model_name+'.pkl'

    clf = joblib.load(file_name)


    features = ['Date','Price','Root']+eval(features)

    df_copy = df[features].copy()

    prediction = clf.predict(df[features[3:]])
    df_copy['Prediction'] = prediction
    trades = df_copy[df_copy['Prediction']>=model['cutoff']]
    trades['Profit'] = None
    trades['SPY Profit'] = None
    for trade in trades.iterrows():
        key, value = trade
        start_date = datetime.strptime(str(value['Date'])[:-2],'%Y%m%d')
        end_date = (start_date + timedelta(days=91)).strftime('%Y%m%d')

        new_data = coll.find({'Root': value['Root'], 'Date': {'$gte': int(end_date)}},{'Date': 1,'Price': 1}).sort('Date', pymongo.ASCENDING)

        spy_start = spy.find({'Root': 'SPY', 'Update_Date': {'$gte': str(start_date.strftime('%Y%m%d'))}},{'Update_Date': 1,'Underlying_Price': 1}).sort('Update_Date', pymongo.ASCENDING).limit(1)
        spy_end = spy.find({'Root': 'SPY', 'Update_Date': {'$lte': str(end_date)}},{'Update_Date': 1,'Underlying_Price': 1}).sort('Update_Date', pymongo.DESCENDING).limit(1)


        try:
            start_price = float(value['Price'])
            close_price = float(pd.DataFrame(list(new_data)).head(1)['Price'])
            profit = (close_price - start_price)/start_price
            spy_start_price = float(pd.DataFrame(list(spy_start)).head(1)['Underlying_Price'])
            spy_close_price = float(pd.DataFrame(list(spy_end)).head(1)['Underlying_Price'])
            spy_profit = (spy_close_price - spy_start_price)/spy_start_price


            trades.loc[key,'Profit'] = float(profit)
            trades.loc[key,'SPY Profit'] = float(spy_profit)
        except:
            continue

    #trades = trades.dropna(subset=['Profit'])
    #print(trades.tail(1))
    print(model_name, len(trades), round(trades['Profit'].mean(),4), round(trades['SPY Profit'].mean(),4))
    trades.to_csv('trades.csv')
