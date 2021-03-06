import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.preprocessing import StandardScaler
init_notebook_mode(connected=True)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier
from sklearn import svm

# import lightgbm as lgb
# from lightgbm import LGBMClassifier
# import xgboost as xgb

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

data_df = pd.read_csv('/Users/mac/Downloads/个人欺诈/creditcard.csv')
print("Credit Card Fraud Detection data -  rows:", data_df.shape[0], " columns:", data_df.shape[1])

data_df.head()
# data_df.describe().to_csv(r'/home/lynette/creditcardfraud/dataset/imbalanced_describe.csv')

temp = data_df['Class'].value_counts()
df = pd.DataFrame({'Class': temp.index, 'values': temp.values})
# 只有492 (0.172%)个交易是欺诈

data_df['Minute'] = (data_df['Time'].apply(lambda x: np.floor(x / 60))) % (24 * 60)
# print(data_df.describe())

# —————————————————— time amount 标准化 ———————————————————————————

data_df['std_Amount'] = StandardScaler().fit_transform(data_df['Amount'].values.reshape(-1, 1))
data_df['std_Minute'] = StandardScaler().fit_transform(data_df['Minute'].values.reshape(-1, 1))
# —————————————————— data_df ———————————————————————————

data_df = data_df
data_df.drop('Time', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列
data_df.drop('Amount', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列
data_df.drop('Minute', axis=1, inplace=True)  # axis参数默认为0表示删除行，1是删除列

print("原始样本集data_df -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
# print(data_df.describe())

fraud_data = data_df[data_df["Class"] == 1]  # fraud_data dataframe
number_records_fraud = len(fraud_data)  # how many fraud samples:492
not_fraud_data = data_df[data_df["Class"] == 0].sample(n=number_records_fraud)
data_df_under = pd.concat([fraud_data, not_fraud_data])
print("降采样的样本集data_df_under -  rows:", data_df_under.shape[0], " columns:", data_df_under.shape[1])





import matplotlib.pyplot as plt

labels=['正常','欺诈']
X=[284315,492]

fig = plt.figure()
plt.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("数据集类别占比图")

plt.show()





plt.figure(figsize=(14, 14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="Blues")
plt.show()

# V1-V28之间没有显着的相关性。时间与V3呈反相关，数量与V7和V20成正相关，数量与V1和V5成反相关
# class=1 欺诈
#
# # ———————— 把相关变量画图
# #
# s = sns.lmplot(x='V20', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# s = sns.lmplot(x='V7', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# plt.show()
# #
# # # 确认数量和V7 V20 正相关的（Class = 0的回归线具有正斜率，而Class = 1的回归线具有较小的正斜率）。
# #
# s = sns.lmplot(x='V2', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# s = sns.lmplot(x='V5', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# plt.show()
#
# # 确认数量和V2 V5 负相关的（Class = 0的回归线具有正斜率，而Class = 1的回归线具有较小的正斜率）。
#
# ———————————————————————————— 特征密度图
# var = data_df.columns.values
#
# i = 0
# t0 = data_df.loc[data_df['Class'] == 0]
# t1 = data_df.loc[data_df['Class'] == 1]
#
# sns.set_style('whitegrid')
# plt.figure()
# fig, ax = plt.subplots(8, 4, figsize=(16, 28))
#
# for feature in var:
#     i += 1
#     plt.subplot(8, 4, i)
#     sns.kdeplot(t0[feature], bw=0.5, label="Class = 0")
#     sns.kdeplot(t1[feature], bw=0.5, label="Class = 1")
#     plt.xlabel(feature, fontsize=12)
#     locs, labels = plt.xticks()
#     plt.tick_params(axis='both', which='major', labelsize=12)
# plt.show()
#
# plt.close()

# V4，V11区别很大，V12，V14，V18有些区别，V1，V2，V3 ，V10部分区别，而V25，V26，V28表现类似。

# 除了时间和金额，合法交易的特征分布（Class = 0的值）以0为中心，有时在其中一个末端排队很长。而欺诈性交易（Class的值= 1）不对称分布。

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# —————— 预测模型

#
# target = 'Class'
# predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#               'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
#               'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
#               'Amount']
#
# train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
# train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)
# # RandomForrestClassifier
#
# print(train_df.shape[0])
# print(test_df.shape[0])
# print(valid_df.shape[0])
#
# clf = RandomForestClassifier(n_jobs=NO_JOBS,
#                              random_state=RANDOM_STATE,
#                              criterion=RFC_METRIC,
#                              n_estimators=NUM_ESTIMATORS,
#                              verbose=False)
#
# print(clf)
#
# clf.fit(train_df[predictors], train_df[target].values)
# preds = clf.predict(valid_df[predictors])

# print(preds)

#
# # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
# tmp = tmp.sort_values(by='Feature importance', ascending=False)
# plt.figure(figsize=(7, 4))
# plt.title('Features importance', fontsize=14)
# s = sns.barplot(x='Feature', y='Feature importance', data=tmp)
# s.set_xticklabels(s.get_xticklabels(), rotation=90)
# plt.show()
#
# # V17, V12, V14, V10, V11, V16最重要
#
# cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
# fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
# sns.heatmap(cm, xticklabels=['Not Fraud', 'Fraud'],
#             yticklabels=['Not Fraud', 'Fraud'],
#             annot=True, ax=ax1,
#             linewidths=.2, linecolor="Darkblue", cmap="Blues")
# plt.title('Confusion Matrix', fontsize=14)
# plt.show()

# roc_auc_score(valid_df[target].values, preds)
# print(roc_auc_score(valid_df[target].values, preds))



