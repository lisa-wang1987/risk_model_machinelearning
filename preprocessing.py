# import api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.interpolate import lagrange
import scorecard_functions_V3 as sf
import datetime
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import pickle
import itertools

import warnings
warnings.filterwarnings("ignore")



#preprocessing function

#home_ownership turn 'ANY' to 'MORTGAGE'
def any_to_mort(val):
    if val=='ANY':
        return 'MORTGAGE'
    else:
        return val
#delete '%' string function
def int_rate(val):
    if val != 'nan':
        return round(float(str(val).replace('%',''))/100,4)
    elif val == str('nan'):
        return -1
#delete‘year’string function
def emp_length(val):
    if val =='10+ years':
        return 10
    elif val =='< 1 year':
        return 0
    elif val == 'n/a':
        return -1
    elif val =='1 year':
        return 1
    else:
        return float(str(val).replace('years',''))

#date transformation function
def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
        #time.mktime 不能读取1970年之前的日期
    else:
        yr = int(x[4:6])
        if yr <=17:
            yr = 2000+yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr,mth,1)

#date to date long function
def days_long(val):
    now = datetime.datetime.now()
    delta = now - val
    return delta.days


#grade
def grade_value(val):
    grade = val.replace('A',1)
    grade = grade.replace('B',2)
    grade = grade.replace('C',3)
    grade = grade.replace('D',4)
    grade = grade.replace('E',5)
    grade = grade.replace('F',6)
    grade = grade.replace('G',7)
    return grade

#subgrade
def subgrade_value(val):
    grade = val.replace('A1',11)
    grade = grade.replace('A2',12)
    grade = grade.replace('A3',13)
    grade = grade.replace('A4',14)
    grade = grade.replace('A5',15)
    grade = grade.replace('B1',21)
    grade = grade.replace('B2',22)
    grade = grade.replace('B3',23)
    grade = grade.replace('B4',24)
    grade = grade.replace('B5',25)
    grade = grade.replace('C1',31)
    grade = grade.replace('C2',32)
    grade = grade.replace('C3',33)
    grade = grade.replace('C4',34)
    grade = grade.replace('C5',35)
    grade = grade.replace('D1',41)
    grade = grade.replace('D2',42)
    grade = grade.replace('D3',43)
    grade = grade.replace('D4',44)
    grade = grade.replace('D5',45)
    grade = grade.replace('E1',51)
    grade = grade.replace('E2',52)
    grade = grade.replace('E3',53)
    grade = grade.replace('E4',54)
    grade = grade.replace('E5',55)
    grade = grade.replace('F1',61)
    grade = grade.replace('F2',62)
    grade = grade.replace('F3',63)
    grade = grade.replace('F4',64)
    grade = grade.replace('F5',65)
    grade = grade.replace('G1',71)
    grade = grade.replace('G2',72)
    grade = grade.replace('G3',73)
    grade = grade.replace('G4',74)
    grade = grade.replace('G5',75)
    return grade

# missing fill function
def mean_val(data,col):
    mean_col = data[col].mean()
    idx = data[data[col].isnull()==True].index
    data.loc[idx,col]=mean_col
    return data[col]

#label transformation function
def label_transe(val):
    if val == 'Charged Off':
        return 1
    elif (val == 'Fully Paid' or val =='Current'):
        return 0
    else:
        return -1

# read data and print statistic information
def read_data(data):
    print('The data is the third quarter of 2017 borrower data of LendingClub opened on official website')
    df = pd.read_csv(data,header=1)
    df = df[df['term']==' 36 months']
    print ('\n')
    print('top 5 line of data is :\n',df.head(2))
    print('\n')
    print('data statistic information is ',df.describe())
    print('\n')
    print('all data shape is:',df.shape)
    return df

# separate training set and test set
def split_train_test(data):
    trainData = data[(data['issue_d']!='Nov-2015') & (data['issue_d']!='Dec-2015')]
    testData = data[(data['issue_d']=='Nov-2015') | (data['issue_d']=='Dec-2015')]
    print('the train data shape is:',trainData.shape)
    print('the test data shape is:',testData.shape)
    return trainData,testData

#data preprocessing
def drop_afterloan_columns(data):
    #drop after loan feature
    after_col =['pymnt_plan', 'collection_recovery_fee', 'recoveries', 'hardship_flag','title',
            'out_prncp_inv', 'out_prncp','total_rec_prncp','last_pymnt_amnt','last_pymnt_d',
            'last_credit_pull_d','total_pymnt','total_pymnt_inv','total_rec_int',
            'total_rec_late_fee','title','term']
    df = data.drop(after_col,axis=1)
    print('after drop after loan data, the data shape is ',df.shape)
    return df

# drop only one value columns
def drop_unique1_col(data):
    cols =data.nunique()[data.nunique()>1].index.tolist()
    df =data.loc[:,cols]
    print('after drop only one value columns, the data shape is ',df.shape)
    return df

# drop columns if it's missing greater than 60%
def drop_missingmore60_col(data):
    miss_60_col = data.isnull().sum()[data.isnull().sum()>=0.40*data.shape[0]].index
    df = data.drop(miss_60_col,axis=1)
    print('after drop missing greater than 60% columns, the data shape is ',df.shape)
    return df

#drop all null row and all null column
def drop_row_col_miss(data):
    data = data.dropna(how='all',axis=1)
    df = data.dropna(how='all',axis=0)
    return df

#delete 90% value same in one column
def drop_90samevalue_col(data):
    colum=data.columns
    per=pd.DataFrame(colum,index=colum)
    max_valuecounts=[]
    for col in colum:
        max_valuecounts.append(data[col].value_counts().max())
    per['mode']=max_valuecounts
    per['percentil'] =per['mode']/data.shape[0]
    same_value_col =per[per.sort_values(by='percentil',ascending=False)['percentil']>0.9].index
    df = data.drop(same_value_col,axis=1)
    print('after delete 90% values same in one column,the data shape is',df.shape)
    return df

#get label function
def get_label(data,label):
    data['y'] = data[label].apply(label_transe)
    data =data[((data['y']!=2) & (data['y']!=-1))].drop([label],axis=1)
    return data

#character to value
def string_to_value(data):
    data['int_rate'] = data.loc[:,'int_rate'].apply(int_rate) #int_rate
    data['emp_length'] = data.loc[:,'emp_length'].apply(emp_length) # emp_length
    data['revol_util']=data.loc[:,'revol_util'].astype(str).apply(int_rate) # revol_util
    data['grade'] = grade_value(data.loc[:,'grade'])#grade
    data['sub_grade'] = subgrade_value(data.loc[:,'sub_grade'])#sub_grade
    data['earliest_cr_line'] = data.loc[:,'earliest_cr_line'].apply(ConvertDateStr).apply(days_long)
    return data


#get catogery columns and continues columns
def class_feature(data):
    cat_col = list(data.columns.to_series().groupby(data.dtypes).groups.values())[2]
    continue_col = list(data.columns.to_series().groupby(data.dtypes).groups.values())[0].append(list(data.columns.to_series().groupby(data.dtypes).groups.values())[1])

    return cat_col,continue_col

#get word_col,cat_col,ordered_col,continue_col
def get_word_cat_ordered_continue_col(data):
    # text feature columns
    word_col=['zip_code', 'addr_state','emp_title']
    temp_col=['mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mths_since_recent_bc',
             'mths_since_recent_inq','pct_tl_nvr_dlq','percent_bc_gt_75','bc_util','revol_util','dti']
    cat_col,value_col = class_feature(data)
    cat_col = [w for w in cat_col if w not in word_col] #catergory columns
    continue_col = []
    for col in value_col:
        if len(set(data[col]))>500:
            continue_col.append(col)
    continue_col =[key for key in continue_col if key not in word_col]
    continue_col = [key for key in continue_col if key !='emp_length']
    continue_col = [key for key in continue_col if key not in temp_col] #continue col

    ordered_col = [key for key in value_col if key not in continue_col]
    ordered_col = [key for key in ordered_col if key !='y']
    ordered_col = [key for key in ordered_col if key !='term']
    ordered_co = ordered_col + ['emp_length'] # ordered columns
    return word_col,cat_col,ordered_col,continue_col


# filling columns has missing value
def missing_fill(data):
    cat_col,continue_col = class_feature(data) # 区分分类特征和连续特征
    missing_col=list(data.isnull().sum()[data.isnull().sum()>0].index)
    for col in missing_col:
        if col in cat_col:
            fill_value = data[col].mode()[0]
            data[col]=data[col].fillna(fill_value)
        else:
            fill_value = data[col].mean()
            data[col] = data[col].fillna(fill_value)
    return data

# outlier
#del orded columns and continues top 10 greater samples
def del_outlier_index(data,key):
    temp_index = data[data[key]>data[key].sort_values(ascending=False)[0:10].min()].index
    return list(temp_index)
