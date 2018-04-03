import pandas as pd
import gc



def f_template(df, gb_dict):
    
    all_features = list(set(gb_dict['groupby'] + [gb_dict['select']]))
    ## name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'], gb_dict['select'])
    ## perfom the grouby
    gp = df[all_features]. \
        groupby(gb_dict['groupby'])[gb_dict['select']]. \
        agg(gb_dict['agg']). \
        reset_index(). \
        rename(index=str, columns={gb_dict['select']: new_feature}).astype(gb_dict['type'])
    # Merge back to df
    df = df.merge(gp, on=gb_dict['groupby'], how='left')
    del gp
    gc.collect()

    return df, df.columns.values.tolist()

def f_base(df):
    df.drop(['attributed_time', 'is_attributed'], axis=1, inplace=True)
    gc.collect()
    return df, df.columns.values.tolist()


def f_1(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    return df, df.columns.values.tolist()

def f_1_2(df):
    df['click_time'] = pd.to_datetime(df.click_time)
    df['click_rnd']=df['click_time'].dt.round('H')
    df['hour'] = pd.to_datetime(df.click_rnd).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_rnd).dt.day.astype('uint8')
    df.drop(['click_time', 'click_rnd'], axis=1, inplace=True)
    gc.collect()
    return df, df.columns.values.tolist()

def f_2(df):
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
    return df, df.columns.values.tolist()

def f_c_1_1(df):
    # Count, for ip
    gb_dict = {'groupby': ['ip'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_1_2(df):
    # Count, for app
    gb_dict = {'groupby': ['app'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_1_3(df):
    # Count, for device
    gb_dict = {'groupby': ['device'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_1_4(df):
    # Count, for os
    gb_dict = {'groupby': ['os'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_1_5(df):
    # Count, for channel
    gb_dict = {'groupby': ['channel'], 'select': 'os', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_2_1(df):
    # Count, for ip-app
    gb_dict = {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}  
    return f_template(df, gb_dict)   

def f_c_2_2(df):
    # Count, for ip-day
    gb_dict = {'groupby': ['ip','day'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_2_3(df):
    # Count, for ip-hour
    gb_dict = {'groupby': ['ip','hour'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_3_1(df):
    # Count, for ip-day-hour
    gb_dict = {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_3_2(df):
    # Count, for ip-app-os
    gb_dict = {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

def f_c_4_1(df):
    # Count, for ip-app-day-hour
    gb_dict = {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    return f_template(df, gb_dict)

    # # Mean hour, for ip-app-channel
    # {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean', 'type': 'float32', 'type': 'float32'}

    # # Variance in day, for ip-app-channel
    # {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var', 'type': 'float32'}
    # # Variance in day, for ip-app-device
    # {'groupby': ['ip','app','device'], 'select': 'day', 'agg': 'var', 'type': 'float32'}
    # # Variance in day, for ip-app-os
    # {'groupby': ['ip','app','os'], 'select': 'day', 'agg': 'var', 'type': 'float32'}

    # # Variance in hour, for ip-app-channel
    # {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'var'}
    # # Variance in hour, for ip-app-device
    # {'groupby': ['ip','app','device'], 'select': 'hour', 'agg': 'var'}
    # # Variance in hour, for ip-app-os
    # {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'}




if __name__ == "__main__":
    train_df = pd.read_csv('../input/train.csv', nrows=1000)

    ## test f_base
    train_df, df_set = f_base(train_df)
    assert df_set == ['ip', 'app', 'device', 'os', 'channel', 'click_time']

    ## test f_1
    train_df, df_set = f_1_2(train_df)
    assert df_set == ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']

    ## test f_2
    train_df, df_set = f_2(train_df)
    assert df_set == ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day', 'nip_day_test_hh']

    ## test f_template
    gb_dict1 = {'groupby': ['ip'], 'select': 'channel', 'agg': 'count', 'type': 'float32', 'type': 'float32'}
    gb_dict2 = {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'mean', 'type': 'uint32'}

    train_df, df_set = f_template(train_df, gb_dict1)
    assert df_set == ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day', 'nip_day_test_hh', \
                      'ip_count_channel']

    train_df, df_set = f_template(train_df, gb_dict2)
    assert df_set == ['ip', 'app', 'device', 'os', 'channel', 'hour', \
                      'day', 'nip_day_test_hh', 'ip_count_channel','ip_app_mean_channel']

    


    
