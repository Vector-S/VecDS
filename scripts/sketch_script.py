import pandas as pd



data = pd.read_csv('../output/submission.csv')
data.reset_index(inplace=True)
data.rename(columns = {'index':'click_id'},inplace=True)
pass
data.to_csv('../output/submission_0420.csv',float_format='%.8f',index=False)


