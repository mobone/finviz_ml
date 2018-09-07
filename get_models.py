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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

experiments = 6
max_mean = 0
class Machine(threading.Thread):
    def __init__(self, df, q, thread_id):
        threading.Thread.__init__(self)
        self.df = df
        self.q = q
        self.thread_id = thread_id
        self.max_mean = 0


    def run(self):
        self.conn = sqlite3.connect('results.db')
        print('starting thread')
        while q.qsize():

            features, dividend_toggle = q.get()


            df = self.df.copy()

            if dividend_toggle:
                df = df[df['Dividend %']>0.0]

            targets = df[df.columns[-4:]]
            df = df[df.columns[:-4]]

            ml_models = []


            df = df[list(features)]
            bottom_cutoff = []
            result_saves = []
            for i in range(experiments):
                X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size=0.33)


                clf = SVR(C=1.0, epsilon=0.2)
                clf.fit(X_train, y_train['Percent Change'])
                predictions = pd.Series(clf.predict(X_test),index=y_test.index)
                predictions.name = 'Predictions'

                result = pd.concat([y_test, predictions],axis=1)

                result = result.sort_values(by='Predictions')
                result_saves.append(result.copy())

                bottom_cutoff.append(result.tail(50).head(1)['Predictions'].values[0])

                ml_model = result.tail(50)
                if dividend_toggle:
                    ml_model['Dividend %'] = ml_model['Dividend %'].fillna(0)
                    ml_model['Percent Change'] = ml_model['Percent Change'] + (ml_model['Dividend %']/4)
                ml_models.append(ml_model['Percent Change'])


            ml_models = pd.concat(ml_models)
            result_saves = pd.concat(result_saves)
            #result_saves.to_csv('test.csv')

            ml_stats = ml_models.describe()

            bottom_cutoff = sum(bottom_cutoff)/len(bottom_cutoff)


            if ml_stats['mean']>(self.max_mean*.95) and ml_stats['mean']>.1:

                self.max_mean = ml_stats['mean']

                ml_interval = mean_confidence_interval(ml_models)

                row = [str(list(X_test.columns)),ml_interval[0],ml_interval[1],ml_interval[2],bottom_cutoff,dividend_toggle]


                details = pd.Series(row,index=['Features','mean', 'mean-','mean+','cutoff','dividend_toggle'])
                details = pd.DataFrame(details,columns=['Details']).T

                clf = SVR(C=1.0, epsilon=0.2)

                clf.fit(df[df.columns[:-4]],targets['Percent Change'])
                name = str(features)[1:-1]
                print(name)
                name = name.replace(', ','_').replace("'",'').replace('/','-o-')

                joblib.dump(clf, 'models/%s %s.pkl' % (ml_stats['mean'].round(4), name))
                print(details.T)
                details.to_sql('model_results',self.conn,if_exists='append')
                print(name)
                print('===================')


def clean_data(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    df = df.drop(df.columns[0],1)

    return df

def get_features(df):


    targets = df[['Root','Date','Percent Change','Dividend %']]

    df = df.drop(['Root','Date','Percent Change','Dividend %'],1)

    imp = Imputer(missing_values='NaN', strategy='mean')
    imp.fit(df)
    data = pd.DataFrame(imp.transform(df),columns=df.columns)


    KBest = SelectKBest(k=12)
    KBest = KBest.fit(data, targets['Percent Change'])

    features = list(data.columns[KBest.get_support()])
    data = data[features]



    df = pd.concat([data, targets],axis=1)


    return features, df

df = pd.read_csv('comapany_data.csv')

df['Percent Change'] = (df['Close Price'] - df['Price']) / df['Price']

df = df.drop(['Ticker', 'Shortable', '52W Range', 'Close Price', 'Price'], 1)

#machine(df)
df = clean_data(df)
features, df = get_features(df)



feature_choices = []
for permute_length in range(4,7):
    feature_choices.extend(list(itertools.permutations(features, r=permute_length)))

shuffle(feature_choices)
input_choices = []
q = queue.Queue()

for item in feature_choices:
    #q.put((item, True))
    q.put((item, False))



for i in range(17):
    x = Machine(df, q, i)
    x.start()
