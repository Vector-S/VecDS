import time
import pickle
import json
import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics
from matplotlib import pyplot
import gc






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
MAX_TRAIN_ROWS = 184903889
MAX_TEST_ROWS =  18790468
NUM_TRAIN_ROWS=100000
NUM_TEST_ROWS=None
SKIP_TRAIN_ROWS=range(1,MAX_TRAIN_ROWS-NUM_TRAIN_ROWS)

FEATURE_PPL = [f_base,f_hour,f_count,f_mean_hour]
METHOD = 'xgb'
TRANSDUCTIVE = False
PARA_TUNE = False
RELEASE = True
INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'
PARA_TUNE_CFG = '../cfg/default.cfg'

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

if PARA_TUNE:
    while(1):
        s.para_tune()
        report("Choose your option.")
        option = input("\n\nDo you want to continue tuning parameter ?[Y/N]\n")
        if option=='N':break
        else:
            while(1):
                try:
                    s.para_tune_fcg = input("\nPlease specify a new cfg file:\n")
                    cfg = load_json(s.para_tune_fcg)
                    print("To double check, your cfg file is:\n{0}\n{1}".format(s.para_tune_fcg,cfg))
                    confirm = input("\nPlease confirm [Y/N]:\n")
                except Exception as e:
                    continue
                if confirm=='Y':break
if RELEASE:
    s.train()
    s.test()
    s.save_test()







