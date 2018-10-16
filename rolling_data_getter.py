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


today = datetime.now() + timedelta(days=1)
oldest_alerts = today - timedelta(days=180)
#start = today - timedelta(days=180)
start = today - timedelta(days=2)
total_dataset = []
prev_day = None
for i in range(int(start.strftime('%Y%m%d')), int(today.strftime('%Y%m%d'))):
    print(i)
    try:
        today = datetime.strptime(str(i),'%Y%m%d')
        if today in NYSE_holidays() or today.isoweekday() not in range(1,6):
            continue
    except:
        continue

    start = time.time()

    todays = coll.find({'Date': i},{'_id':0,'_rev':0,'Ticker':0}).sort([('Root', pymongo.ASCENDING)])
    todays = pd.DataFrame(list(todays))


    if todays.empty:
        continue

    todays = todays.set_index(todays['Root'])
    #todays = todays.drop('Root',1)

    end = time.time()
    print(end-start)


    todays = todays.dropna(subset=['Sales','Earnings'])


    if prev_day is not None:
        #prev_day = prev_day.dropna()
        # TODO: ignore single day changes, or, changes that aren't near earnings date

        mask = todays.Sales.ne(prev_day.Sales)
        total_dataset.append(todays[mask])




        out_df = pd.concat(total_dataset)
        out_df['Root'] = list(out_df.index)
        out_df = out_df.reset_index(drop='True')
        out_df['Days till Earnings'] = None
        for row in out_df.iterrows():
            key, value = row[0], row[1]

            if value['Earnings'][-3:] == 'AMC' or value['Earnings'][-3:] == 'BMO':
                earnings_dates = [value['Earnings'][:-4] + ' 2017', value['Earnings'][:-4] + ' 2018']
            else:
                earnings_dates = [value['Earnings'] + ' 2017', value['Earnings'] + ' 2018']

            current_date = datetime.strptime(str(value['Date'])[:-2], '%Y%m%d')
            earnings_dates[0] = datetime.strptime(earnings_dates[0], '%b %d %Y')
            earnings_dates[1] = datetime.strptime(earnings_dates[1], '%b %d %Y')
            if abs((current_date-earnings_dates[0]).days)<=7:
                out_df.ix[key,'Days till Earnings'] = abs((current_date-earnings_dates[0]).days)
            elif abs((current_date-earnings_dates[1]).days)<=7:
                out_df.ix[key,'Days till Earnings'] = abs((current_date-earnings_dates[1]).days)

        print(out_df)
        print('----------')
        out_df = out_df.dropna(subset=['Days till Earnings'])

        out_df = out_df.drop_duplicates(subset=['Root','Sales'])

        #out_df.to_csv('rolling_data.csv')
        print(out_df)
        current_alerts = pd.read_csv('rolling_data.csv')
        current_alerts = current_alerts[current_alerts['Date']>=float(oldest_alerts.strftime('%Y%m%d'))]
        print(current_alerts)

        current_alerts.append(out_df)
        current_alerts.to_csv('rolling_data.csv', index=False)


        #print(float(prev_day['Date'].head(1)))
        print(float(todays['Date'].head(1)))
        #print(out_df[out_df['Root']=='AAPL'].ix[:,['Date','Sales','Earnings']])
        print('============')
    prev_day = todays
    #print(mask)
