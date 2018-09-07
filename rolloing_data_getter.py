from pymongo import MongoClient
import pymongo
import pandas as pd
import configparser
from datetime import datetime, timedelta
import threading
import queue
from datetime import datetime
import time
from nyse_holidays import NYSE_holidays
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

total_dataset = []
prev_day = None
for i in range(20171031,20181006):
    try:
        today = datetime.strptime(str(i),'%Y%m%d')
        if today in NYSE_holidays() or today.isoweekday() not in range(1,6):
            continue
    except:
        continue

    start = time.time()

    todays = coll.find({'Date': i}).sort([('Root', pymongo.ASCENDING)])
    todays = pd.DataFrame(list(todays))

    if todays.empty:
        continue

    todays = todays.set_index(todays['Root'])
    #todays = todays.drop('Root',1)

    end = time.time()
    print(end-start)


    todays = todays.dropna(subset=['Sales'])

    if prev_day is not None:
        prev_day = prev_day.dropna(subset=['Sales'])
        # TODO: ignore single day changes, or, changes that aren't near earnings date
        mask = todays.Sales.ne(prev_day.Sales)
        total_dataset.append(todays[mask])

        out_df = pd.concat(total_dataset)

        out_df = out_df.drop_duplicates(subset=['Root','Sales'])

        out_df.to_csv('rolling_data.csv')
        print(len(out_df))
        print(float(prev_day['Date'].head(1)))
        print(float(todays['Date'].head(1)))
        print(out_df[out_df['Root']=='AAPL'].ix[:,['Date','Sales','Earnings']])
        print('============')
    prev_day = todays
    #print(mask)
