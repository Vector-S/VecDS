import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # for validation
import gc  # memory
from datetime import datetime  # train time checking
import time
import sys
sys.path.append("/Users/ruixuezhang/Desktop/KaggleTAFDC")

########################################## Internal Lib #####################################
# when releasing, move all functions used into main script
from lib.utils import *
from lib.featurelib import *
from lib.modellib import *
########################################## Macro Control Panel #####################################

VALIDATE=True
SKIP_ROWS=range(1,109903891)
TRAIN_ROWS=5000
TEST_ROWS=100

FEATURE_PPL = [f_base,f_1,f_1_2,f_2]
METHOD = 'xgb'
##########################################         Path           #####################################



########################################## Solution Class #################################################

class Solution:
    output_filename = 'submission.csv'
    input_path = '../input/'
    output_path = '../output/'
    train_file = 'train.csv'
    test_file = 'test.csv'

    def __init__(self):
        train_df = None
        label_df = None
        test_df = None
        f_ppl = []
        train_f_set = None
        test_f_set =None
        model = None
        train_result = None
        test_result = None
        log= None
        method = None

    def load_data(self):
        train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
        test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
        dtypes = {
            'ip': 'uint32',
            'app': 'uint16',
            'device': 'uint16',
            'os': 'uint16',
            'channel': 'uint16',
            'is_attributed': 'uint8',
            'click_id': 'uint32'
        }
        self.train_df = pd.read_csv(self.input_path+self.train_file, skiprows=SKIP_ROWS, nrows=TRAIN_ROWS, dtype=dtypes,
                               usecols=train_cols)
        self.test_df = pd.read_csv(self.input_path +self.test_file, dtype=dtypes, nrows=TEST_ROWS, usecols=test_cols)
        self.label_df = self.train_df['is_attributed']

    def build_features(self):
        self.train_df,self.train_f_set = build_features(self.train_df,self.f_ppl)
        self.test_df,self.test_f_set = build_features(self.test_df,self.f_ppl)
        print("Feature Selected:\t{0}".format(','.join(self.train_f_set)))

    def init_model(self):
        if self.method == 'xgb':
            xgb_init(self)
        elif self.method == 'lgbm':
            pass

    def para_tune(self):
        if self.method == 'xgb':
            xgb_pt_1(self)
        elif self.method == 'lgbm':
            pass

    def train(self):
        if self.method == 'xgb':
            xgb_train(self)
            xgb_save_fi(self)
        elif self.method == 'lgbm':
            pass

    def test(self):
        if self.method == 'xgb':
            xgb_test(self)
        elif self.method == 'lgbm':
            pass

    def save_test(self):
        self.test_result.to_csv(self.output_path + self.output_filename, float_format='%.8f', index=False)




########################################## Solution Excution #################################################
s = Solution()
s.method = METHOD

report("Load Dataset Start")
tic = time.time()
s.load_data()
report("Load Dataset Done",tic)


tic = time.time()
report("Build Features Start")
s.f_ppl= FEATURE_PPL
s.build_features()
report("Build Features Done",tic)



tic=time.time()
report("Model training Start")
s.init_model()
s.para_tune()
s.train()
report("Model training Done",tic)


tic=time.time()
report("Test Start")
s.test()
report("Test Done",tic)


s.save_test()
report("Output Saved",tic)
pass







