import pandas as pd
import scorecard_functions_V3 as sf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
import visualization as vs
#get catergory col
def get_less_more_cat_col(data,cols_set):
    less_col = []
    more_col =[]
    for col in cols_set:
        if len(set(data[col])) <=5:
            less_col.append(col)
        else:
            more_col.append(col)
    return less_col,more_col

#define dummy code for less cat col
def dummy_code(data,less_cat_col):
    return pd.concat([data,pd.get_dummies(data[less_cat_col])],axis=1).drop(less_cat_col,axis=1)

#get more col code
def get_more_cat_code(data,col_set,label):
    for col in col_set:
        data[col+'_Bin'] = data[col].map(sf.BinBadRate(data,col,label)[0])
    return data
# get order columns cutoff points and new data

# order merge and bin code
'''
i explore found,some order feature not only have some value no bad sample and
some no good sample,so here processing them separately.
'''
def order_merge_encode(data,order_col):# order_col is sure
    '''
    param data: input data
    param order_col: all order features need to be deal with
    '''
    merge_bin_dict_bad={}
    merge_bin_dict_good={}
    order_list_bad=[]
    order_list_good=[]

    # test bad part
    print('starting bad part=========')
    for col in order_col:
        binBadRate = sf.BinBadRate(data, col, 'y')[0]
        if min(binBadRate.values()) == 0 :  #由于某个取值没有坏样本而进行合并
            print ('{} need to be combined due to 0 bad rate'.format(col))
            combine_bin = sf.MergeBad0(data, col, 'y')
            merge_bin_dict_bad[col] = combine_bin
            newVar = col + '_Merge'
            order_list_bad.append(newVar)
            data[newVar] = data[col].map(combine_bin)
    del_list_bad=[w.replace('_Merge','') for w in order_list_bad]
    data = data.drop(del_list_bad,axis=1)
    colmns_set= [w for w in data.columns if w != 'y']

    # test good part
    print('starting good part==========')
    for col in colmns_set:
        binBadRate = sf.BinBadRate(data, col, 'y')[0]
        if min(binBadRate.values()) == 1 :  #由于某个取值没有坏样本而进行合并
            print ('{} need to be combined due to 0 good rate'.format(col))
            combine_bin = sf.MergeBad0(data, col, 'y')
            merge_bin_dict_good[col] = combine_bin
            newVar = col + '_Merge'
            order_list_good.append(newVar)
            data[newVar] = data[col].map(combine_bin)
    del_list_good=[w.replace('_Merge','') for w in order_list_good]
    data = data.drop(del_list_good,axis=1)
    return data

# order encode
def order_encode(data,order_col):
    '''
    param data: input data
    param order_col: all order features need to be deal with
    '''
    df = order_merge_encode(data,order_col)#use above function
    list_set = [w for w in df.columns if w !='y']
    df_data = get_more_cat_code(df,list_set,'y')
    df_data = df_data.drop(list_set,axis=1)
    return df_data

# get cutoff point and get new data
def get_cutoff(data,cols_set,label):
    less_cols,more_cols = get_less_more_cat_col(data,cols_set)
    merge_bin_dict={}

    continues_merged_dict={}
    var_bin_list =[]
    for col in less_cols:
        binBadRate = sf.BinBadRate(data, col, 'y')[0]
        if min(binBadRate.values()) == 0 :  #由于某个取值没有坏样本而进行合并
            print ('{} need to be combined due to 0 bad rate'.format(col))
            combine_bin = sf.MergeBad0(data, col, 'y')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            data[newVar] = data[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:    #由于某个取值没有好样本而进行合并
            print ('{} need to be combined due to 0 good rate'.format(col))
            combine_bin = sf.MergeBad0(data, col, 'y',direction = 'good')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            data[newVar] = data[col].map(combine_bin)
            order_list.append(newVar)
            var_bin_list.append(newVar)

    for col in more_cols:
        print('{} is in processing'.format(col))
        max_interval = 5
        cutoff = sf.ChiMerge(data,col,label,max_interval=max_interval,minBinPcnt=0)
        data[col+'_Bin'] = data[col].map(lambda x: sf.AssignBin(x,cutoff,special_attribute=[]))
        monotone = sf.BadRateMonotone(data,col+'_Bin',label)
        while (not monotone):
            max_interval -=1
            cutoff = sf.ChiMerge(data,col,label,max_interval=max_interval,special_attribute=[],minBinPcnt=0)
            data[col +'_Bin'] = data[col].map(lambda x: sf.AssignBin(x,cutoff,special_attribute=[]))
            if max_interval == 2:
                break
            monotone = sf.BadRateMonotone(data,col+'_Bin',label)
        newVar = col +'_Bin'
        data[newVar] = data[col].map(lambda x: sf.AssignBin(x,cutoff,special_attribute=[]))
        var_bin_list.append(newVar)
        continues_merged_dict[col] = cutoff
    return continues_merged_dict,var_bin_list,data

# get woe and iv value
def get_woe_iv(data,cols_set,label):
    col_iv ={}
    col_woe={}
    for col in cols_set:
        temp = sf.CalcWOE(data,col,label)
        col_iv[col] = temp['IV']
        col_woe[col] = temp['WOE']
    return col_iv,col_woe

#get top iv>0.01
def choose_iv_feature(data,col_iv,col_woe):
    high_IV = {k:v for k, v in col_iv.items() if v >= 0.01}
    high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)

    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var + '_WOE'
        data[newVar] = data[var].map(col_woe[var])
        short_list_2.append(newVar)
    return short_list_2


# correlation
def cor_feature(data,col_iv):
    deleted_index = []
    high_IV = {k:v for k, v in col_iv.items() if v >= 0.01}
    high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0]+"_WOE"
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0]+"_WOE"
            roh = np.corrcoef(data[x1],data[y1])[0,1]
            if abs(roh)>0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)
    multi_analysis_vars_1 = [high_IV_sorted[i][0]+"_WOE" for i in range(cnt_vars) if i not in deleted_index]
    return multi_analysis_vars_1

# train model

# calculate ks value
def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)

#lr training
def lr_train(X,y,clf):
    clf.fit(X,y)
    pred = clf.predict_proba(X)[:,1]
    df = pd.DataFrame([pred,y]).T
    df.columns = ['score','y']
    print('the roc curve is:')
    vs.model_roc_curve(X,y,clf)
    print('the ks value is :',KS(df,'score','y'))


# grid research
def grid_research(X,y,clf,paremeter):
    from sklearn.model_selection import GridSearchCV
    cv = StratifiedKFold(n_splits=5)
    clf = RandomForestClassifier()
    grid = GridSearchCV(clf,paremeter,cv=cv)
    grid.fit(X,y)
    return grid.best_params_
