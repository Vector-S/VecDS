import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # for validation
import gc  # memory
from datetime import datetime  # train time checking
import time
import sys
sys.path.append("/Users/ruixuezhang/Desktop/KaggleTAFDC")

########################################## Internal Lib #####################################
# when releasing, move all functions used into main script

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



################################################# XGBoost #################################################

def xgb_init(s):
    s.model =  XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99)




def modelfit(model, train_df, label_df, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        dtrain = xgb.DMatrix(train_df.values, label=label_df.values)
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        model.set_params(n_estimators=cvresult.shape[0])
        del dtrain, cvresult
        gc.collect()
    # Fit the algorithm on the data
    model.fit(train_df, label_df, eval_metric='auc')

def xgb_train(s,eval_metric='auc'):
    modelfit(s.model,s.train_df,s.label_df)


    s.train_result = s.model.predict_proba(s.train_df)[:,1]
    print("\nModel Report")
    print("AUC Score (Train):", metrics.roc_auc_score(s.label_df.values, s.train_result))


def xgb_save_fi(s):
    plot_importance(s.model)
    pyplot.savefig(s.output_path+ 'feature_importance.png')

def xgb_test(s):
    s.test_result = pd.DataFrame()
    s.test_result['is_attributed'] = s.model.predict_proba(s.test_df)[:,1]
    s.test_result['click_id']=s.test_result.index


################################ LightGBM ############################################

def lgbm_init(s):
    pass

def lgbm_pt(s,cv_folds=5, early_stopping_rounds=50):
    pass

def lgbm_train(s,eval_metric='auc') :
    pass

def lgbm_save_fi(s):
    pass

def lgbm_test(s,test_df):
    pass


def report(msg,tic=None,print_out=True):
    length = 100
    report = "\n"+">>>>  "+ msg + " "
    if tic:
        report += '-' * (100 - len(report))
        report +=" TC:{0:.4g}".format(time.time()-tic)
    report+="\n"
    if print_out:
        print(report)
    if not tic:
        return time.time()


def load_obj(filename):
    try:
        with open(filename,'rb') as f:
            obj =  pickle.load(f)
            return obj
    except Exception as e:
        print("Can't load obj:{0}".format(str(e)))

def save_obj(obj,filename):
    try:
        with open(filename,'wb') as f:
            pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print("Can't save obj:{0}".format(str(e)))


def load_json(filename):
    try:
        with open(filename,'r') as f:
            dic =  json.load(f)
        return dic
    except Exception as e:
        print("Can't load dict:{0}".format(str(e)))

def save_json(dic,filename):
    try:
        with open(filename,'w') as f:
            json.dump(dic,f)
    except Exception as e:
        print("Can't save dict:{0}".format(str(e)))


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class Report:
    def __init__(self):
        total_time_cost = 0




def build_features(df, feature_pipeline):
    feature_set = set()
    for fun in feature_pipeline:
        df, feature_set = fun(df, feature_set)
    df = df[list(feature_set)]
    gc.collect()
    return df, feature_set


def f_template(df, gb_dict):
    all_features = list(set(gb_dict['groupby'] + [gb_dict['select']]))
    ## name of new feature
    if gb_dict['agg'] == 'count':
        f_name = '{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'])
    else:
        f_name = '{}_{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'], gb_dict['select'])
    ## perfom the grouby
    gp = df[all_features]. \
        groupby(gb_dict['groupby'])[gb_dict['select']]. \
        agg(gb_dict['agg']). \
        reset_index(). \
        rename(index=str, columns={gb_dict['select']: f_name}).astype(gb_dict['type'])
    # Merge back to df
    df = df.merge(gp, on=gb_dict['groupby'], how='left')
    del gp
    gc.collect()
    return df, f_name


#######################  feature definition zone #######################
def f_base(df, fs):
    fs = fs | {'ip', 'app', 'device', 'os', 'channel'}
    return df, fs


def f_hour(df, fs):
    f_name = 'hour'
    df[f_name] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    gc.collect()
    fs.add(f_name)
    return df, fs


def f_dfw(df, fs):
    f_name = 'dayofweek'
    df[f_name] = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')
    gc.collect()
    fs.add(f_name)
    return df, fs


def f_count(df, fs):
    groupby_list = [['ip'], ['os', 'device'], ['os', 'device', 'hour'], ['app', 'channel'], ['ip', 'hour']]
    select = 'click_time'
    agg = 'count'
    type = 'uint32'
    for groupby in groupby_list:
        gb_dict = {'groupby': groupby, 'select': select, 'agg': agg, 'type': type}
        df, f_name = f_template(df, gb_dict)
        fs.add(f_name)
    return df, fs


def f_mean_hour(df, fs):
    # Count, for ip
    groupby_list = [['ip'], ['os', 'device']]
    select = 'hour'
    agg = 'mean'
    type = 'uint32'
    for groupby in groupby_list:
        gb_dict = {'groupby': groupby, 'select': select, 'agg': agg, 'type': type}
        df, f_name = f_template(df, gb_dict)
        fs.add(f_name)
    fs.add(f_name)
    return df, fs


def f_2(df, fs):
    """
    :param df:
    :param fs:
    :return:
    """
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip', 'day', 'in_test_hh'], how='left')
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    del gp
    gc.collect()
    fs |= {'nip_day_test_hh'}
    return df, fs


def f_1_2(df, fs):
    click_time = pd.to_datetime(df.click_time)
    df['click_rnd'] = click_time.dt.round('H')
    df['hour'] = pd.to_datetime(df.click_rnd).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_rnd).dt.day.astype('uint8')
    df.drop('click_rnd', axis=1, inplace=True)
    fs = fs | {'hour', 'day'}
    gc.collect()
    return df, fs



class DataSet:
    train_file = 'train.csv'
    test_file = 'test.csv'
    label_name = 'is_attributed'
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    skip_train_rows = None
    num_train_rows = None
    num_test_rows = None
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }
    def __init__(self):
        pass

    def load_dataset(self,s):
        s.train_df = pd.read_csv(s.input_path+self.train_file, skiprows=self.skip_train_rows, nrows=self.num_train_rows, dtype=self.dtypes,
                               usecols=self.train_cols)
        s.test_df = pd.read_csv(s.input_path +self.test_file, dtype=self.dtypes, nrows=self.num_test_rows, usecols=self.test_cols)
        s.label_df = s.train_df[self.label_name]


class Solution:
    train_df = None
    label_df = None
    test_df = None

    f_ppl = []
    train_f_set = None
    test_f_set = None

    model = None
    train_result = None
    test_result = None

    log = None
    method = 'xgb'
    result_filename = 'submission.csv'
    input_path = '../input/'
    output_path = '../output/'
    data_set = None
    transductive =False
    para_tune_fcg = None

    def __init__(self):
        pass

    def load_dataset(self):
        tic = report("Load Dataset Start")
        self.data_set.load_dataset(self)
        report("Load Dataset Done", tic)

    def build_features(self):
        tic = report("Build Features Start")
        if self.transductive:
            merge = pd.concat([self.train_df, self.test_df])
            merge, f_set = build_features(merge, self.f_ppl)
            self.train_f_set,self.test_f_set = f_set,f_set
            self.train_df, self.test_df = merge[:self.train_df.shape[0]],merge[self.train_df.shape[0]:].reset_index(drop=True)
        else:
            self.train_df, self.train_f_set = build_features(self.train_df, self.f_ppl)
            self.test_df, self.test_f_set = build_features(self.test_df, self.f_ppl)
            if self.train_f_set !=self.test_f_set:
                print("Warning training featue set is different with testing feature set")

        print("Feature Selected:\t{0}".format(','.join(self.train_f_set)))
        report("Build Features Done", tic)
        print(self.train_df.head(5))
        print(self.test_df.head(5))

    def init_model(self):
        if self.method == 'xgb':
            xgb_init(self)
        elif self.method == 'lgbm':
            pass

    def para_tune(self):
        tic = report("Parameter Tuning Start")
        if self.method == 'xgb':
            xgb_pt(self)
        elif self.method == 'lgbm':
            pass
        report("Parameter Tuning Done", tic)

    def train(self):
        tic = report("Model training Start")
        if self.method == 'xgb':
            xgb_train(self)
            xgb_save_fi(self)
        elif self.method == 'lgbm':
            pass
        report("Model training Done", tic)

    def test(self):
        tic = report("Test Start")
        if self.method == 'xgb':
            xgb_test(self)
        elif self.method == 'lgbm':
            pass
        report("Test Done", tic)

    def save_test(self):
        check_dir(self.output_path)
        self.test_result.to_csv(self.output_path + self.result_filename, float_format='%.8f', index=False)




def xgb_pt(s):
    """
    to be implemented
    :param s:
    :return:
    """
    pass




########################################## Macro Control Panel #####################################
TRAIN_FILE = 'train.csv'
TEST_FILE= 'test.csv'
SKIP_TRAIN_ROWS=range(1,109903891)
NUM_TRAIN_ROWS=10000
NUM_TEST_ROWS=None
FEATURE_PPL = [f_base,f_hour,f_dfw,f_count,f_mean_hour]
METHOD = 'xgb'
TRANSDUCTIVE = False
PARA_TUNE = False
RELEASE = True
INPUT_PATH = '../input/'
OUTPUT_PATH = './'

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







