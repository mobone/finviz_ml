
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
symbols = coll.find({},{'Root': 1}).distinct('Root')
out_df = []
for symbol in symbols:
    data = coll.find({'Root':symbol},{'_id':0,'_rev':0,'Ticker':0})
    groups = pd.DataFrame(list(data)).groupby('Sales')
    for group in groups:

        out_df.append(group[1].head(1))
    print(pd.concat(out_df))
