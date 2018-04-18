import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # for validation
import gc  # memory
from datetime import datetime  # train time checking
import time
import sys
sys.path.append("/Users/ruixuezhang/Desktop/KaggleTAFDC")

########################################## Internal Lib #####################################
# when releasing, move all functions used into main script
from veclib.solution import *
from veclib.modellib import *
from veclib.ptlib import *
from veclib.featurelib import *
from veclib.utils import *
########################################## Macro Control Panel #####################################
TRAIN_FILE = 'train.csv'
TEST_FILE= 'test.csv'
SKIP_TRAIN_ROWS=range(1,109903891)
NUM_TRAIN_ROWS=5000
NUM_TEST_ROWS=100
FEATURE_PPL = [f_base,f_hour,f_dfw,f_count,f_mean_hour]
METHOD = 'xgb'
TRANSDUCTIVE = False
PARA_TUNE = False
RELEASE = True
INPUT_PATH = './input/'
OUTPUT_PATH = './output/'
PARA_TUNE_CFG = './cfg/default.cfg'

##########################################         Setting up           #####################################

d = DataSet()
d.train_file=TRAIN_FILE
d.test_file=TEST_FILE
d.skip_train_rows=SKIP_TRAIN_ROWS
d.num_train_rows = NUM_TRAIN_ROWS
d.num_test_rows= NUM_TEST_ROWS

s = Solution()
s.data_set = d
s.method = METHOD
s.f_ppl= FEATURE_PPL
s.para_tune_fcg = PARA_TUNE_CFG
s.transductive = TRANSDUCTIVE
s.input_path =INPUT_PATH
s.output_path=OUTPUT_PATH
##########################################            Excution                ##################################

s.load_dataset()
s.build_features()
s.init_model()

if RELEASE:
    s.train()
    s.test()
    s.save_test()







