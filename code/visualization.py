# ignore warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import helper as lp

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib')

#Display inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from numpy.random import beta
import seaborn as sns
plt.style.use('bmh')


#plot borrower loan status distribution
def plot_label(data,col):
    temp = data.groupby('y').count().iloc[:,0]
    bar_data = {'normal':temp[0],'overdue':temp[1]}
    names = list(bar_data.keys())
    values = list(bar_data.values())
    plt.bar(range(2),values)
    plt.xticks((0,1),('normal','overdue'))
    plt.title('borrower loan status distribution')
    plt.text(0.45,180000,r'normal:overdue mostly equal to 7',color='black')
    plt.text(0.45,170000,r'unbalanced dataset',color='black')
    plt.show()

#plot category columns <5
def plot_cat(data,key):
    plot_data=data[[key,'y']]
    plt.figure(figsize=(8,6))
    if (key=='home_ownership'):
        values=['ANY','RENT','MORTGAGE','OWN']
    if (key=='verification_status'):
        values=['Source Verified', 'Not Verified', 'Verified']
    if (key=='initial_list_status'):
        values=['w', 'f']
    if (key=='grade'):
        values=[1, 2, 3, 4, 5, 6, 7]
    if (key=='emp_length'):
        values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.027167861525544, 8.0, 9.0, 10.0, 7.0]
    if (key=='purpose'):
        values=['home_improvement', 'medical', 'educational', 'other', 'debt_consolidation', 'vacation', 'house', 'wedding',
                'major_purchase', 'moving', 'car', 'small_business', 'renewable_energy', 'credit_card']
    if (key=='issue_d'):
        values=['Oct-2015', 'Dec-2015', 'Aug-2015', 'Apr-2015', 'May-2015', 'Nov-2015', 'Jan-2015', 'Sep-2015',
                'Jun-2015', 'Feb-2015', 'Mar-2015', 'Jul-2015']
    #create DataFrame containing categories and each of counts
    frame = pd.DataFrame(index=np.arange(len(values)),columns=(key,'normal','overdue'))
    for i,value in enumerate(values):
        frame.loc[i]=[value,len(plot_data[(plot_data['y'] == 0) & (plot_data[key] == value)]),len(plot_data[(plot_data['y'] == 1) & (plot_data[key] == value)])]
    #display each categrory's overdue rate
    bin_width = 0.4
    for i in np.arange(frame.shape[0]):
        overdue_bar = plt.bar(i-bin_width,frame.loc[i]['overdue'],width = bin_width,color='r')
        normal_bar = plt.bar(i,frame.loc[i]['normal'],width = bin_width,color='g')

        plt.xticks(np.arange(len(frame)),values)
        plt.legend((overdue_bar[0],normal_bar[0]),('overdue','normal'),framealpha=0.8)

    plt.xlabel(key)
    plt.ylabel('number of borrower')
    plt.xticks(rotation=90)
    plt.title('borrower Statistics With \'%s\' Feature'%(key))
    plt.show()


    # Report number of passengers with missing values
    if sum(pd.isnull(plot_data[key])):
        nan_outcomes = plot_data[pd.isnull(plot_data[key])]['y']
        print ("borrower with missing '{}' values: {} ({} overdue, {} normal)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))

#plot word cloud
def word_cloud(data,key):
    word_freq=Counter()
    word = data[key].astype(str)
    word_freq =Counter(word)
    words_cloud = WordCloud(scale=5,min_font_size=8,max_words=100,background_color='white').fit_words(word_freq)
    plt.imshow(words_cloud)

    # Report number of passengers with missing values
    if sum(pd.isnull(data[key])):
        nan_outcomes = data[pd.isnull(data[key])]['y']
        print ("borrower with missing '{}' values: {} ({} overdue, {} normal)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))



#plot youxu columns
def plot_youxu_col(data,label,key):
    all_data=data[[key,label]]
    all_data = all_data[~np.isnan(all_data[key])]
    plt.figure(figsize=(8,6))
    min_value = all_data[key].min()
    max_value = all_data[key].max()
    value_range = max_value - min_value
    overdue = all_data[all_data['y']==1][key]
    normal = all_data[all_data['y']==0][key]
    bins = np.arange(min_value-1,all_data[key].max()+1,1)
    plt.hist(overdue,bins=bins,histtype='stepfilled',alpha=0.6,color='red',label='overdue')
    plt.hist(normal,bins=bins,histtype='stepfilled',alpha=0.6,color='green',label='normal')
    plt.xlim(0,bins.max())
    plt.legend(framealpha=0.8)
    plt.xlabel(key)
    plt.ylabel('Number of borrower')
    plt.title('borrower overdue Statistics With \'%s\' Feature'%(key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print ("borrower with missing '{}' values: {} ({} overdue, {} normal)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))

#plot lianxu columns distribution
def plot_lianxu_col(data,label,key):
    all_data=data[[key,label]]
    plt.figure(figsize=(8,6))
    min_value = all_data[key].min()
    max_value = all_data[key].max()
    value_range = max_value - min_value
    overdue = all_data[all_data['y']==1][key]
    normal = all_data[all_data['y']==0][key]
    if (key=='installment'):
        bins = np.arange(min_value-1,all_data[key].max()+1,100)
    else:
        bins = np.arange(min_value-1,all_data[key].max()+1,1000)
    plt.hist(overdue,bins=bins,histtype='stepfilled',alpha=0.6,color='red',label='overdue')
    plt.hist(normal,bins=bins,histtype='stepfilled',alpha=0.6,color='green',label='normal')
    plt.xlim(0,bins.max())
    plt.legend(framealpha=0.8)
    plt.xlabel(key)
    plt.ylabel('Number of borrower')
    plt.title('borrower overdue Statistics With \'%s\' Feature'%(key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['y']
        print ("borrower with missing '{}' values: {} ({} overdue, {} normal)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))

# plot iv sort bar
def feature_IV_bar(col_IV):
    IV_dict_sorted = sorted(col_IV.items(),key=lambda x:x[1],reverse=True)
    IV_values =[i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.title('feature IV value bar')
    plt.bar(range(len(IV_values)),IV_values)
    plt.show()




# define cross validation roc curve and split use 5
def model_roc_curve(X,y,clf):
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

#corr heatmap
def plot_corrmatrix_heatmap(data):
    df_data = data.corr()
    plt.subplots(figsize=(9,9))
    sns.heatmap(df_data,annot = False,vmax=1, square=True, cmap="Blues")
    plt.show()


#plot sorted iv value bar
def iv_sorted(col_iv):
    IV_dict_sorted = sorted(col_iv.items(),key=lambda x:x[1],reverse=True)
    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.title('sorted feature IV bar')
    #plt.bar(range(len(IV_values)),IV_values)
    plt.bar(IV_name,IV_values)
    plt.show()

#plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#plot ks curve
def ks_curve(pred_y_data):
    total = pred_y_data.groupby(['score'])['y'].count()
    bad = pred_y_data.groupby(['score'])['y'].sum()
    alls = pd.DataFrame({'total':total,'bad':bad})
    alls['good'] = alls['total'] - alls['bad']
    alls['score'] = alls.index
    alls = alls.sorted_values(by='score',ascending=False)
    alls.index = range(len(alls))
    alls['badCumRate'] = alls['bad'].cumsum() / alls['bad'].sum()
    alls['goodCumRate'] = alls['good'].cumsum() / alls['good'].sum()
    plt.plot(alls['badCumRate'])
    plt.plot(alls['goodCumRate'])
    plt.show()
