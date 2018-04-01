import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # for validation
import gc  # memory
from datetime import datetime  # train time checking

start = datetime.now()
########################################## macro control panel #####################################
VALIDATE = False
RANDOM_STATE = 50
VALID_SIZE = 0.90
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 650
SKIP_ROWS = range(1, 109903891)
N_ROWS = 75000000

####################################################################################################

output_filename = 'submission.csv'
path = '../input/'
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path + "train.csv", skiprows=SKIP_ROWS, nrows=N_ROWS, dtype=dtypes, usecols=train_cols)
gc.collect()


def build_features(train_df,feature_pipeline):
    feature_set=set()
    for fun in feature_pipeline:
        feature_add,feature_drop=fun(train_df)
        feature_set-=set(feature_drop)
        feature_set|=set(feature_add)
    return train_df,feature_set

from solutions.featurelib import *

f_pipeline = [f_base,f_1,f_2]

train_df,feature_set = build_features(train_df,f_pipeline)

print(feature_set)
print(train_df)