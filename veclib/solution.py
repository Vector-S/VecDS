from veclib.modellib import *
from veclib.ptlib import *
from veclib.featurelib import *
from veclib.utils import *


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


    def load_train(self,s):
        s.train_df = pd.read_csv(s.input_path+self.train_file, skiprows=self.skip_train_rows, nrows=self.num_train_rows, dtype=self.dtypes,
                               usecols=self.train_cols)
        s.label_df = s.train_df[self.label_name]

    def load_test(self,s):
        s.test_df = pd.read_csv(s.input_path +self.test_file, dtype=self.dtypes, nrows=self.num_test_rows, usecols=self.test_cols)



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
        if self.transductive:
            self.data_set.load_train(self)
            self.data_set.load_test(self)
        else:
            self.data_set.load_test(self)
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



