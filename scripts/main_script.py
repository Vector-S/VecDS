import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # for validation
import gc  # memory
from datetime import datetime  # train time checking
import time

########################################## Internal Lib #####################################
# when releasing, move all functions used into main script
from lib.utils import *
from lib.featurelib import *
from lib.modellib import *
########################################## Macro Control Panel #####################################

VALIDATE=False,
SKIP_ROWS=range(1,109903891)
TRAIN_ROWS=1000
TEST_ROWS=1000


##########################################         Path           #####################################

output_filename = 'submission.csv'
input_path = '../input/'
output_path = '../output/'

########################################## Common Info #################################################

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
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

####################################################################################################

def load_dataset():

    train_df = pd.read_csv(input_path + "train.csv", skiprows=SKIP_ROWS, nrows=TRAIN_ROWS, dtype=dtypes, usecols=train_cols)
    test_df = pd.read_csv(input_path + "test.csv", dtype=dtypes, nrows=TEST_ROWS, usecols =test_cols)

    gc.collect()
    return train_df,test_df

tic = time.time()
report("Load Dataset Start")
train_df,test_df = load_dataset()
report("Load Dataset Done",tic)


label_df = train_df['is_attributed']
print("Train data size: {0}".format(train_df.shape))
print("Test data size: {0}".format(test_df.shape))

train_df.drop('is_attributed',axis=1,inplace=True)



def build_features(df, feature_pipeline):
    feature_set=set()
    for fun in feature_pipeline:
        df, feature_set=fun(df, feature_set)
    df = df[list(feature_set)]
    gc.collect()
    return df, feature_set

f_pipeline = [f_base,f_1,f_1_2,f_2]
report("Build Features Start")

tic = time.time()
train_df,feature_set = build_features(train_df,f_pipeline)
test_df,feature_set = build_features(test_df,f_pipeline)
report("Build Features Done",tic)
print("Feature Selected:\t{0}".format(','.join(feature_set)))

pass


tic=time.time()
report("Model training Start")
model = xgb_train(train_df, label_df, validate=VALIDATE)
report("Model training Done",tic)


tic=time.time()
report("Prediction Start")
prediction = xgb_predict(model,test_df)
report("Prediction Done",tic)


prediction.to_csv(output_path + output_filename, float_format='%.8f', index=False)
report("Output Saved",tic)
pass









