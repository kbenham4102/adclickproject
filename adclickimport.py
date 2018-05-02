import numpy as np
import pandas as pd
import gc
from sklearn.utils import shuffle
import random

def getbatch(path, filename):

    columns = ['ip','app','device','os', 'channel', 'hour',
     'day', 'ip_day_hour', 'ip_app_count', 'ip_app_os_count', 'is_attributed']
    train_df = pd.read_csv(path + "/" + filename, usecols = columns)
    m = len(train_df)

    Y = np.array(train_df['is_attributed']).reshape(m, 1)

    train_df.drop(['is_attributed'], 1, inplace = True)
    gc.collect()

    X = np.array(train_df[:m])

    return X, Y

def simple(path, filename):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    columns = ['ip','app','device','os', 'channel', 'click_time']
    train_df = pd.read_csv(path + "/" + filename, dtype = dtypes,
    usecols = columns)


    return train_df


def get_randbatch(percent):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
    filename = 'train.csv'
    path = 'C:/Users/Kevin/scripts/adclickproj/mnt/ssd/kaggle-talkingdata2/competition_files'
    columns = ['ip','app','device','os', 'channel', 'click_time',
     'is_attributed']

    train_df = pd.read_csv(path + "/" + filename, dtype = dtypes,
    usecols = columns, skiprows = lambda i: i>0 and random.random() > percent)

    gc.collect()

    # Add a few features...
    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['hour'] = train_df['click_time'].dt.hour.astype('uint8')
    train_df['day'] = train_df['click_time'].dt.day.astype('uint8')
    m = len(train_df)

    # Group by IP,DAY,HOUR
    n_chans = train_df[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_day_hour'})
    train_df = train_df.merge(n_chans, on=['ip','day','hour'], how='left')
    del n_chans
    gc.collect()

    # Group by IP and APP
    n_chans = train_df[['ip','app', 'channel']].groupby(by=['ip',
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(n_chans, on=['ip','app'], how='left')
    del n_chans
    gc.collect()

    # Group by IP APP OS
    n_chans = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(n_chans, on=['ip','app', 'os'], how='left')
    del n_chans
    gc.collect()

    # Don't need these...
    train_df.drop(['click_time'], 1,inplace = True)
    gc.collect()


    return train_df


def gen_rand_set(path, batchsize):
    percent = 0.270321 #To generate roughly 50,000,000 row set from 184,000,000
    df = get_randbatch(percent)
    df.to_csv(path + '/' + 'master.csv', index = False)
    m = len(df)
    print('Length of training set in total: '+str(m))
    gc.collect()

    # Shuffle and center data
    print("Shuffling the data...")
    df = shuffle(df)
    Y = df['is_attributed']
    df.drop(['is_attributed'], 1, inplace = True)
    gc.collect()

    # Center data and save feature means
    print('Centering the data...')
    means = df.mean()
    df -= means
    means.to_csv(path + '/' + 'feature_means.csv', index = False)
    del means
    gc.collect()

    # Scale by standard deviation
    print('Scaling the data...')
    sig = df.std()
    df /= sig
    sig.to_csv(path + '/' + 'std_devs.csv', index = False)
    del sig
    gc.collect()

    df['is_attributed'] = Y
    print('Dividing into minibatches and saving...')
    for i in range(int(50000000/batchsize)):
        adrs = path + '/' + 'MBcent' + str(i) + '.csv'
        df[i*batchsize:(i+1)*batchsize].to_csv(adrs, index = False)
        if i%100 == 0:
            print('Batch number ' + str(i+1) + ' saved!')
    del df
    gc.collect()
    print('Total number = ' + str(m)  + '\n' + 'No. of Batches = ' + str(i+1))
    return None

def get_testset():
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
    filename = 'test.csv'
    path = 'C:/Users/Kevin/scripts/adclickproj/testdat'
    columns = ['click_id', 'ip','app','device','os', 'channel', 'click_time']

    df = pd.read_csv(path + "/" + filename, dtype = dtypes,
    usecols = columns)

    gc.collect()

    # Add a few features...
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    m = len(df)

    # Group by IP,DAY,HOUR
    n_chans = df[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_day_hour'})
    df = df.merge(n_chans, on=['ip','day','hour'], how='left')
    del n_chans
    gc.collect()

    # Group by IP and APP
    n_chans = df[['ip','app', 'channel']].groupby(by=['ip',
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    df = df.merge(n_chans, on=['ip','app'], how='left')
    del n_chans
    gc.collect()

    # Group by IP APP OS
    n_chans = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
    df = df.merge(n_chans, on=['ip','app', 'os'], how='left')
    del n_chans
    gc.collect()

    # Don't need these...
    df.drop(['click_time', 'click_id'], 1,inplace = True)
    gc.collect()

    # Use data specifically saved combining statistics of test set and train
    path = 'C:/Users/Kevin/scripts/adclickproj/MBcent2/feature_means.csv'
    weighted_mean = np.array(pd.read_csv(path)).reshape(1,10)
    path = 'C:/Users/Kevin/scripts/adclickproj/MBcent2/std_devs.csv'
    pooled_std = np.array(pd.read_csv(path)).reshape(1,10)
    df = np.array(df).astype('float')
    df -= weighted_mean
    df /= pooled_std

    return df

def gen_batches(batchsize):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'hour'          : 'uint16',
            'day'           : 'uint8',
            'ip_day_hour'   : 'uint16',
            'ip_app_count'  : 'uint32',
            'ip_app_os_count': 'uint32'
            }
    filename = 'master.csv'
    path = 'C:/Users/Kevin/scripts/adclickproj/MBcent2'
    columns = ['ip', 'app',	'device', 'os', 'channel', 'is_attributed',
    	'hour',	'day',	'ip_day_hour', 'ip_app_count', 'ip_app_os_count']

    df = pd.read_csv(path + "/" + filename, dtype = dtypes, usecols = columns)

    m = len(df)
    print('Length of training set in total: '+str(m))
    gc.collect()

    # Shuffle and center data
    print("Shuffling the data...")
    df = shuffle(df)
    Y = df['is_attributed']
    df.drop(['is_attributed'], 1, inplace = True)
    gc.collect()

    # Center data and save feature means
    print('Centering the data...')
    means = df.mean()
    df -= means
    means.to_csv(path + '/' + 'feature_means.csv', index = False)
    del means
    gc.collect()

    # Scale by standard deviation
    print('Scaling the data...')
    sig = df.std()
    df /= sig
    sig.to_csv(path + '/' + 'std_devs.csv', index = False)
    del sig
    gc.collect()

    df["is_attributed"] = Y
    print('Dividing into minibatches and saving...')
    for i in range(int(50000000/batchsize)):
        adrs = path + '/' + 'MBcent' + str(i) + '.csv'
        df[i*batchsize:(i+1)*batchsize].to_csv(adrs, index = False)
        if i%100 == 0:
            print('Batch number ' + str(i+1) + ' saved!')
    del df
    gc.collect()
    print('Total number = ' + str(m)  + '\n' + 'No. of Batches = ' + str(i+1))
    return None



'''
path = 'C:/Users/Kevin/scripts/adclickproj'
filename = "train_sample.csv"
 = getbatch(path, filename, 1, 1001, 1000)


print('Shape of data: ', X.shape)
print('Shape of truth: ', Y.shape)
print('Num features = '+str(k))
print('Num examples = '+str(m))
'''
