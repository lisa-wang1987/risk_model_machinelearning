import pickle
import numpy as np
import pandas as pd
import scorecard_functions_V3 as sf
from statsmodels.stats.outliers_influence import variance_inflation_factor



#分类特征个数小于5的分类特征用get_dummy编码
def dummy_code(data,less_cat_col):
    return pd.concat([data,pd.get_dummies(data[less_cat_col])],axis=1).drop(less_cat_col,axis=1)

#分类特征个数大于5的分类特征用badrate编码
#辅助函数
def val_transe(val):
    idx = list(temp.index)
    for i in idx:
        if val == i:
            return m[i]

def bin_count_code(data,col,more_cat_col):
    total = pd.Series(data[col].vaule_counts(),name='total')
    y_1 = pd.Series(data[data[label]==1][col].value_counts(),name='overdue')
    y_0 = pd.Series(data[data[label]==0][col].value_counts(),name='normal')
    counts = pd.DataFrame([y_1,y_0,total]).T
    counts['overdue_rate'] = counts['overdue']/counts['total']
    counts['normal_rate'] = counts['normal']/counts['total']
    counts['log_overdue'] = counts['overdue_rate']/counts['normal_rate']
    return pd.Series(counts['log_overdue'].index)
def get_more_cat_col_code(data,more_cat_col,label):
    for col in more_cat_col:
        temp = bin_count_code(data,col,label)
        data[col + '_encode'] = data[col].apply(val_transe)
        return data

#对于分类个数大于和连续特征进行分箱
def feature_bin(data,label,feature_set):
    #feature_set 需要分箱的特征集合 包括分类和连续
    continues_merged_dict={}
    var_bin_list=[]
    for col in feature_set:
        print('{} is in preprocessing'.format(col))
        max_bin_set = 5
        cutoff = sf.ChiMerge(data,col,label,max_interval=max_bin_set,special_attribute=[],minBinPcnt=0)
        data[col + '_Bin'] = data[col].map(lambda x:sf.AssignBin(x, cutOff, special_attribute=[]))
        monotone = sf.BadRateMonotone(data,col+'_Bin',label,['Bin_1'])
        while (not monotone):
            max_bin_set -=1
            cutoff = sf.ChiMerge(data,col,label,max_interval=max_bin_set,special_attribute=[],minBinPcnt=0)
            data[col + '_Bin'] = data[col].map(lambda x:sf.AssignBin(x, cutOff, special_attribute=[]))
            if max_bin_set ==3:
                break
            monotone = sf.BadRateMonotone(data,col+'_Bin',label,['Bin_1'])
        newvar = col + '_Bin'
        data[newvar] = data[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
        var_bin_list.append(newvar)
    continues_merged_dict[col]=cutoff
    return continues_merged_dict,var_bin_list,data

#分箱完需要计算woe
def IV_WOE(data,label):
    col_IV ={}
    col_woe ={}
    bin_col = [s for s in data.columns if 'Bin' in s]
    for col in bin_col:
        IV_woe = sf.CalcWOE(data,col,label)
        col_IV[col] = IV_woe['IV']
        col_woe[col] = IV_woe['WOE']
        return col_IV,col_woe

#选择iv值>0.01的特征
def feature_select_iv(data,col_IV):
    high_IV = {k:v for k,v in col_IV.items() if v >0.01}
    high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)
    short_list=high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newvar = var +'_WOE'
        data[newvar] = data[var].map(col_woe[var])
        short_list_2.append(newvar)
        return data,short_list_2,high_IV_sorted


#计算两两间的相关系数,若线性相关提出IV值低的
def cal_corr_del_IV_low(data,high_IV_sorted):
    deleted_index=[]
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0]+'_WOE'
        for j in range(cnt_vars):
            if i==j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0]+'_WOE'
            roh = np.corrcoef(data[x1],data[y1])[0,1]
            if abs(roh) >0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)
    multi_analysis_vars = [high_IV_sorted[i][0]+'_WOE' for i in range(cnt_vars) if i not in deleted_index]
    return multi_analysis_vars

#vif诊断
def get_vif(data,cols_set):
    vif={}
    for var in feature_V_L_cor_col.columns:
        new_vif =variance_inflation_factor(feature_V_L_cor_col.values,feature_V_L_cor_col.columns.get_loc(var))
        vif[var]=new_vif
        vif_sorted = pd.Series(list(vif.values()),index=vif.keys())
        return vif_sorted
