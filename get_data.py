from pymongo import MongoClient
import pandas as pd
import configparser
from datetime import datetime, timedelta
import threading
import queue

class Company_getter(threading.Thread):
    def __init__(self, symbol_queue, output_queue):
        threading.Thread.__init__(self)
        self.symbol_queue = symbol_queue
        self.output_queue = output_queue

    def run(self):
        while self.symbol_queue.qsize():
            self.symbol = self.symbol_queue.get()

            self.earnings_dates = self.get_earnings_dates()

            self.earnings_dataframe = self.get_earnings_dataframe()

            self.earnings_dataframe = self.clean_data()

            if not self.earnings_dataframe.empty:
                self.output_queue.put(self.earnings_dataframe)

    def clean_data(self):

        for index, row in self.earnings_dataframe.iterrows():
            for key,value in row.iteritems():
                try:
                    if float(value[:-1]):
                        pass
                    if type(value) is str and value[-1:] == 'K':

                        value = float(value[:-1]) * 1000
                    elif type(value) is str and value[-1:] == 'M':
                        value = float(value[:-1]) * 1000000
                    elif type(value) is str and value[-1:] == 'B':
                        value = float(value[:-1]) * 1000000000
                except:
                    pass

                if type(value) is str and value[-1:] == '%':
                    value = float(value[:-1])

                self.earnings_dataframe.ix[index,key] = value
        print(self.earnings_dataframe.columns)
        input()
        for column in ['_id','_rev','Earnings','Sector','Industry','Optionable','Index','Ticker','Shortable']:
            if column in self.earnings_dataframe.columns:
                self.earnings_dataframe = self.earnings_dataframe.drop(column, axis=1)

        return self.earnings_dataframe

    def get_earnings_dataframe(self):
        prev_sales = None
        prev_income = None
        entries = []
        for earnings_date in self.earnings_dates:
            try:
                sales, income, earnings = self.get_earnings(earnings_date)
            except Exception as e:
                continue
            if prev_sales is None:
                prev_sales = sales
                prev_income = income
            elif sales == prev_sales or income == prev_income:
                prev_sales = sales
                prev_income = income
                continue
            #print(sales, income, earnings_date)
            prev_sales = sales
            prev_income = income
            entries.append(earnings)
        df = pd.DataFrame(entries)
        return df



    def get_earnings_dates(self):

        entries = coll.find({'Root': self.symbol}).sort('Date')
        earnings_dates = []
        for entry in entries:


            read_date = str(int(entry['Date']))
            if 'Earnings' in entry.keys():

                if entry['Earnings'][-3:] == 'AMC' or entry['Earnings'][-3:] == 'BMO':
                    earnings_date = entry['Earnings'][:-4]
                else:
                    earnings_date = entry['Earnings']
            else:
                continue
            if read_date[:4] == '2017':
                earnings_date = earnings_date + ' 2017'
            if read_date[:4] == '2018':
                earnings_date = earnings_date + ' 2018'
            #print(earnings_date)
            earnings_date = datetime.strptime(earnings_date, '%b %d	%Y').strftime('%Y%m%d')
            earnings_dates.append(int(earnings_date))

        earnings_dates = list(set(earnings_dates))
        earnings_dates.sort()
        #print(earnings_dates)
        return earnings_dates

    def get_earnings(self, read_date):
        # get entry details
        entry = coll.find_one({'Root': self.symbol, 'Date': {'$gt': read_date, '$lte':read_date+7}})

        # get close price
        read_date = datetime.strptime(str(read_date), '%Y%m%d')
        read_date = read_date + timedelta(days=60)
        read_date = int(read_date.strftime('%Y%m%d'))
        closing_entry = coll.find_one({'Root': self.symbol, 'Date': {'$gt': read_date, '$lte':read_date+7}}, {'Price':1})
        entry['Close Price'] = closing_entry['Price']

        return entry['Sales'], entry['Income'], entry



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
all_data = []

symbol_queue = queue.Queue()
output_queue = queue.Queue()
for symbol in symbols:
    symbol_queue.put(symbol)

for i in range(10):
    x = Company_getter(symbol_queue, output_queue)
    x.start()

while symbol_queue.qsize():
    while True:
        earnings_dataframe = output_queue.get()

        if not earnings_dataframe.empty:

            all_data.append(earnings_dataframe)

            df = pd.concat(all_data)
            print(df)
            df.to_csv('comapany_data.csv')
