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
########################################## Macro Control Panel #####################################

SKIP_TRAIN_ROWS=range(1,109903891)
NUM_TRAIN_ROWS=5000
NUM_TEST_ROWS=100
FEATURE_PPL = [f_base,f_1,f_1_2,f_2]
METHOD = 'xgb'
PARA_TUNE = False
RELEASE = True
PARA_TUNE_CFG = './cfg/default.cfg'

##########################################         Setting up           #####################################

d = DataSet()
d.skip_train_rows=SKIP_TRAIN_ROWS
d.num_train_rows = NUM_TRAIN_ROWS
d.num_test_rows= NUM_TEST_ROWS

s = Solution()
s.data_set = d
s.method = METHOD
s.f_ppl= FEATURE_PPL
s.para_tune_fcg = PARA_TUNE_CFG

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







