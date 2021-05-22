import pandas as pd
import numpy as np
from matplotlib import gridspec

np.set_printoptions(threshold=np.inf)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm

import lightgbm as lgb
from lightgbm import LGBMClassifier
# import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 100)

RFC_METRIC = 'gini'  # metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100  # number of estimators used for RandomForrestClassifier
NO_JOBS = 4  # number of parallel jobs used for RandomForrestClassifier

# TRAIN/VALIDATION/TEST SPLIT
# VALIDATION
VALID_SIZE = 0.20  # simple validation using train_test_split
TEST_SIZE = 0.20  # test size using_train_test_split

# CROSS-VALIDATION
NUMBER_KFOLDS = 5  # number of KFolds for cross-validation

RANDOM_STATE = 2018

MAX_ROUNDS = 1000  # lgb iterations
EARLY_STOP = 50  # lgb early stop
OPT_ROUNDS = 1000  # To be adjusted based on best validation rounds
VERBOSE_EVAL = 50  # Print out metric result

IS_LOCAL = False

data_df = pd.read_csv('/home/lynette/creditcardfraud/dataset/creditcard.csv')
print("Credit Card Fraud Detection data -  rows:", data_df.shape[0], " columns:", data_df.shape[1])

data_df.head()
# data_df.describe().to_csv(r'/home/lynette/creditcardfraud/dataset/imbalanced_describe.csv')

temp = data_df['Class'].value_counts()
df = pd.DataFrame({'Class': temp.index, 'values': temp.values})
# 只有492 (0.172%)个交易是欺诈

data_df['Minute'] = (data_df['Time'].apply(lambda x: np.floor(x / 60))) % (24 * 60)
print(data_df.describe())

# —————————————————— 时间交易量分布图 ——————————————————————————————
# tmp = data_df.groupby(['Minute', 'Class'])['Amount'].aggregate(
#     ['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
# df = pd.DataFrame(tmp)
# df.columns = ['Minute', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
# # print(df.head())
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
# s = sns.lineplot(ax=ax1, x="Minute", y="Sum", data=df.loc[df.Class == 0])
# s = sns.lineplot(ax=ax2, x="Minute", y="Sum", data=df.loc[df.Class == 1], color="red")
# plt.suptitle("Total Amount")
# plt.show()
# —————————————————— 时间交易量分布图 ——————————————————————————————

# —————————————————— time amount 标准化 ———————————————————————————

data_df['std_Amount'] = StandardScaler().fit_transform(data_df['Amount'].values.reshape(-1, 1))
data_df['std_Minute'] = StandardScaler().fit_transform(data_df['Minute'].values.reshape(-1, 1))

# —————————————————— 其他特征标准化 ———————————————————————————

# features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#             'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
#             'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
#
# for i in features:
#     data_df[i] = StandardScaler().fit_transform(data_df[i].values.reshape(-1, 1))

# —————————————————— data_df ———————————————————————————

data_df = data_df
data_df.drop('Time', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列
data_df.drop('Amount', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列
data_df.drop('Minute', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列

print("data_df -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
print(data_df.describe())

# —————————————————————— 热力图

# # 获取数据
# fraud = data_df[data_df['Class'] == 1]
# nonFraud = data_df[data_df['Class'] == 0]
#
# # 相关性计算
# correlationNonFraud = nonFraud.loc[:, data_df.columns != 'Class'].corr()
# correlationFraud = fraud.loc[:, data_df.columns != 'Class'].corr()
#
# # 上三角矩阵设置
# mask = np.zeros_like(correlationNonFraud)  # 全部设置0
# indices = np.triu_indices_from(correlationNonFraud)  # 返回函数的上三角矩阵
# mask[indices] = True
#
# grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
# f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14, 9))
#
# # 正常用户-特征相关性展示
# cmap = sns.diverging_palette(220, 8, as_cmap=True)
# ax1 = sns.heatmap(correlationNonFraud, ax=ax1, vmin=-1, vmax=1, cmap=cmap, square=False, linewidths=0.5, mask=mask,
#                   cbar=False)
# ax1.set_xticklabels(ax1.get_xticklabels(), size=16)
# ax1.set_yticklabels(ax1.get_yticklabels(), size=16)
# ax1.set_title('Normal', size=20)
#
# # 被欺诈的用户-特征相关性展示
# ax2 = sns.heatmap(correlationFraud, vmin=-1, vmax=1, cmap=cmap,
#                   ax=ax2, square=False, linewidths=0.5, mask=mask, yticklabels=False,
#                   cbar_ax=cbar_ax, cbar_kws={'orientation': 'vertical',
#                                              'ticks': [-1, -0.5, 0, 0.5, 1]})
# ax2.set_xticklabels(ax2.get_xticklabels(), size=16)
# ax2.set_title('Fraud', size=20)

# —————————————————————— V1-V28分析图

# 获取V1-V28 字段

# v_feat_col = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
#               'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
# v_feat_col_size = len(v_feat_col)
#
# plt.figure(figsize=(16, v_feat_col_size * 4))
# gs = gridspec.GridSpec(v_feat_col_size, 1)
# for i, cn in enumerate(data_df[v_feat_col]):
#     ax = plt.subplot(gs[i])
#     sns.displot(data_df[cn][data_df["Class"] == 1], bins=50)  # V1 异常  绿色表示
#     sns.displot(data_df[cn][data_df["Class"] == 0], bins=100)  # V1 正常  橘色表示
#     ax.set_xlabel('')
#     ax.set_title('histogram of feature: ' + str(cn))
#
# plt.savefig('fig.png', bbox_inches='tight')  # 替换 plt.show()

# —————————————————————— 特征重要性排序

# x_feature = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19',
#              'std_Amount', 'std_Minute']
x_feature = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
             'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'std_Amount',
             'std_Minute']

x_val = data_df[x_feature]
y_val = data_df['Class']
clf = RandomForestClassifier(n_estimators=10, random_state=123, max_depth=4)  # 构建分类随机森林分类器
clf.fit(x_val, y_val)  # 对自变量和因变量进行拟合
RandomForestClassifier(max_depth=4, n_estimators=10, random_state=123)

print(clf.feature_importances_)

for feature in zip(x_feature, clf.feature_importances_):
    print(feature)
    print(clf.feature_importances_)

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)

## feature importances 可视化##
importances = clf.feature_importances_
feat_names = data_df[x_feature].columns
indices = np.argsort(importances)[::-1]
fig = plt.figure(figsize=(20, 6))
plt.title("Feature importances by RandomTreeClassifier")

x = list(range(len(indices)))

plt.bar(x, importances[indices], color='lightblue', align="center")
plt.step(x, np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(x, feat_names[indices], rotation='vertical', fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

# ———————————————————————————— 特征密度图

# varr = data_df.columns.values
#
# i = 0
# t0 = data_df.loc[data_df['Class'] == 0]
# # t1 = data_df.loc[data_df['Class'] == 1]
#
# sns.set_style('whitegrid')
# plt.figure()
# fig, ax = plt.subplots(8, 4, figsize=(16, 28))
#
# for feature in varr:
#     i += 1
#     plt.subplot(8, 4, i)
#     sns.kdeplot(t0[feature], bw=0.5, label="Class = 0")
#     # sns.kdeplot(t1[feature], bw=0.5, label="Class = 1")
#     plt.xlabel(feature, fontsize=12)
#     locs, labels = plt.xticks()
#     plt.tick_params(axis='both', which='major', labelsize=12)
# plt.show()

# —————————————————— 决策树

# from sklearn import tree
#
# # 从随机森林抽取单棵树
# estimator = clf.estimators_[5]
#
# #  决策数可视化参考：https://blog.csdn.net/shenfuli/article/details/108492095
# # 导入可视化工具类
# import pydotplus
# from IPython.display import display, Image
#
# # # 注意，根据不同系统安装Graphviz2
# # import os
# # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#
# dot_data = tree.export_graphviz(estimator,
#                                 out_file=None,
#                                 feature_names=x_feature,
#                                 class_names=['0-normal', '1-fraud'],
#                                 filled=True,
#                                 rounded=True
#                                 )
# graph = pydotplus.graph_from_dot_data(dot_data)
# display(Image(graph.create_png()))

#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
#
# parameters = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 3, 10]}
# # C是惩罚力度，与直觉相反，C越小惩罚力度越大
#
# clf = LogisticRegression(n_jobs=-1)
# # model = GridSearchCV(estimator=clf, param_grid=parameters, cv=8, n_jobs=-1, scoring='recall')  # ？？？？？
# # model.fit(train_x_under, train_y_under)
# # print(model.best_score_)
# # print(model.best_params_)
#
# # —————————————————————— 不平衡样本集

# X = data_df.drop(['Class'], axis=1)
# y = data_df['Class']
# print(sorted(Counter(y).items()))
# print("原始样本集data_df -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=33)
#
# clf = LogisticRegression(C=1, penalty='l1', solver='liblinear')
# clf.fit(train_x, train_y)
# predict = clf.predict(test_x)
# print('raw')
# target_names = ['class 0', 'class 1']
# print(classification_report(test_y, predict, target_names=target_names))
# print(confusion_matrix(test_y, predict))

# # —————————————————— data_df_under 降采样的样本集

# fraud_data = data_df[data_df["Class"] == 1]  # fraud_data dataframe
# number_records_fraud = len(fraud_data)  # how many fraud samples:492
# not_fraud_data = data_df[data_df["Class"] == 0].sample(n=number_records_fraud)
# data_df_under = pd.concat([fraud_data, not_fraud_data])
#
# X_under = data_df_under.drop(['Class'], axis=1)
# y_under = data_df_under['Class']
# print(sorted(Counter(y_under).items()))
# print("降采样的样本集data_df_under -  rows:", data_df_under.shape[0], " columns:", data_df_under.shape[1])
# train_x_under, test_x_under, train_y_under, test_y_under = train_test_split(X_under, y_under, test_size=0.3,
#                                                                             random_state=33)
#
# clf = LogisticRegression(C=1, penalty='l1', solver='liblinear')
# clf.fit(train_x_under, train_y_under)
# predict = clf.predict(test_x_under)
# print('under')
# target_names = ['class 0', 'class 1']
# print(classification_report(test_y_under, predict, target_names=target_names))
# print(confusion_matrix(test_y_under, predict))

#
# # # ———————————————————————————————— smote 采样的样本集
#
# X_smote = data_df.drop(['Class'], axis=1)
# y_smote = data_df['Class']
#
# X_smote, y_smote = SMOTE(random_state=33).fit_resample(X_smote, np.ravel(y_smote))
# print(sorted(Counter(y_smote).items()))
# clf = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
# clf.fit(X_smote, y_smote)
# predict = clf.predict(test_x)
# print('smote')
# target_names = ['class 0', 'class 1']
# print(classification_report(test_y, predict, target_names=target_names))
# print(confusion_matrix(test_y, predict))
#
# # # ————————————————————————————————  先under 700：492 再smote 700:700 采样的样本集
#
# fraud_data = data_df[data_df["Class"] == 1]
# number_records_fraud = 700
# not_fraud_data = data_df[data_df["Class"] == 0].sample(n=number_records_fraud)
# data_df_under = pd.concat([fraud_data, not_fraud_data])
#
# X_under = data_df_under.drop(['Class'], axis=1)
# y_under = data_df_under['Class']
# print('先降采样')
# print(sorted(Counter(y_under).items()))
# print("降采样的样本集data_df_under -  rows:", data_df_under.shape[0], " columns:", data_df_under.shape[1])
# train_x_under, test_x_under, train_y_under, test_y_under = train_test_split(X_under, y_under, test_size=0.3, random_state=33)
#
# print('再升采样')
# X_smote, y_smote = SMOTE(random_state=33).fit_resample(X_under, np.ravel(y_under))
# print(sorted(Counter(y_smote).items()))
# clf = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
# clf.fit(X_smote, y_smote)
# predict = clf.predict(test_x)
# print('under+smote')
# target_names = ['class 0', 'class 1']
# print(classification_report(test_y, predict, target_names=target_names))
# print(confusion_matrix(test_y, predict))

# ———————————————————————— 原始随机森林

target = 'Class'
predictors = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
              'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
              'std_Amount', 'std_Minute']

train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)

print(train_df.shape[0])
print(test_df.shape[0])
print(valid_df.shape[0])

clf = RandomForestClassifier(n_jobs=NO_JOBS,
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

print('原始数据集随机森林')
target_names = ['class 0', 'class 1']

clf.fit(train_df[predictors], train_df[target].values)
predict = clf.predict(valid_df[predictors])
print('valid_df')
print(classification_report(valid_df[target], predict, target_names=target_names))
print(confusion_matrix(valid_df[target], predict))
print(roc_auc_score(valid_df[target].values, predict))
predict = clf.predict(test_df[predictors])
print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))

# ———————————————————————— under 随机森林

target = 'Class'
predictors = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
              'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
              'std_Amount', 'std_Minute']

train_df, test_df = train_test_split(data_df_under, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)
# RandomForrestClassifier

print(train_df.shape[0])
print(test_df.shape[0])
print(valid_df.shape[0])

clf = RandomForestClassifier(n_jobs=NO_JOBS,
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
print('under数据集随机森林')
target_names = ['class 0', 'class 1']
clf.fit(train_df[predictors], train_df[target].values)
predict = clf.predict(valid_df[predictors])
print('valid_df')
print(classification_report(valid_df[target], predict, target_names=target_names))
print(confusion_matrix(valid_df[target], predict))
print(roc_auc_score(valid_df[target].values, predict))
predict = clf.predict(test_df[predictors])
print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))

clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                         n_estimators=NUM_ESTIMATORS)
print('under数据集AdaBoostClassifier')
target_names = ['class 0', 'class 1']
clf.fit(train_df[predictors], train_df[target].values)
predict = clf.predict(valid_df[predictors])
print('valid_df')
print(classification_report(valid_df[target], predict, target_names=target_names))
print(confusion_matrix(valid_df[target], predict))
print(roc_auc_score(valid_df[target].values, predict))
predict = clf.predict(test_df[predictors])
print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))

clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                         n_estimators=NUM_ESTIMATORS)
print('under数据集AdaBoostClassifier')
target_names = ['class 0', 'class 1']
clf.fit(train_df[predictors], train_df[target].values)
predict = clf.predict(valid_df[predictors])
print('valid_df')
print(classification_report(valid_df[target], predict, target_names=target_names))
print(confusion_matrix(valid_df[target], predict))
print(roc_auc_score(valid_df[target].values, predict))
predict = clf.predict(test_df[predictors])
print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))

clf = CatBoostClassifier(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='AUC',
                         random_seed=RANDOM_STATE,
                         bagging_temperature=0.2,
                         od_type='Iter',
                         metric_period=VERBOSE_EVAL,
                         od_wait=100)
print('under数据集CatBoostClassifier')
target_names = ['class 0', 'class 1']
clf.fit(train_df[predictors], train_df[target].values)
predict = clf.predict(valid_df[predictors])
print('valid_df')
print(classification_report(valid_df[target], predict, target_names=target_names))
print(confusion_matrix(valid_df[target], predict))
print(roc_auc_score(valid_df[target].values, predict))
predict = clf.predict(test_df[predictors])
print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))



params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.9,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'nthread': 8,
    'verbose': 0,
    'scale_pos_weight': 150,  # because training data is extremely unbalanced
}

dtrain = lgb.Dataset(train_df[predictors].values,
                     label=train_df[target].values,
                     feature_name=predictors)

dvalid = lgb.Dataset(valid_df[predictors].values,
                     label=valid_df[target].values,
                     feature_name=predictors)

evals_results = {}

model = lgb.train(params,
                  dtrain,
                  valid_sets=[dtrain, dvalid],
                  valid_names=['train', 'valid'],
                  evals_result=evals_results,
                  num_boost_round=MAX_ROUNDS,
                  early_stopping_rounds=2 * EARLY_STOP,
                  verbose_eval=VERBOSE_EVAL,
                  feval=None)
predict = model.predict(test_df[predictors])
print(roc_auc_score(test_df[target].values, predict))

print('test_df')
print(classification_report(test_df[target], predict, target_names=target_names))
print(confusion_matrix(test_df[target], predict))
print(roc_auc_score(test_df[target].values, predict))

