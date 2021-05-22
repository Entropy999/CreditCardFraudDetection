# data.describe().to_csv(r'/Users/mac/Downloads/数据集/GiveMeSomeCredit/DataDescribe.csv')


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

init_notebook_mode(connected=True)

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

data_df = pd.read_csv('/home/lynette/creditcardfraud/dataset/creditcard.csv')

print("Credit Card Fraud Detection data -  rows:", data_df.shape[0], " columns:", data_df.shape[1])

# data_df = data_df.iloc[0:100, :]

print("Credit Card Fraud Detection data -  rows:", data_df.shape[0], " columns:", data_df.shape[1])

data_df.head()
# data_df.describe().to_csv(r'/home/lynette/creditcardfraud/dataset/creditcard_describe.csv')

total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum() / data_df.isnull().count() * 100).sort_values(ascending=False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

# no missing data

temp = data_df['Class'].value_counts()
df = pd.DataFrame({'Class': temp.index, 'values': temp.values})

# ——————————————————————————————————————————————————————————————————————————————————————————————
# trace = go.Bar(
#     x=df['Class'], y=df['values'],
#     name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
#     marker=dict(color="Red"),
#     text=df['values']
# )
# data = [trace]
# layout = dict(title='Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
#               xaxis=dict(title='Class', showticklabels=True),
#               yaxis=dict(title='Number of transactions'),
#               hovermode='closest', width=600
#               )
# fig = dict(data=data, layout=layout)
# plot(fig, filename='Class')
# plt.show()

# 只有492 (0.172%)个交易是欺诈 ——————————————————————————————————————————————————————————————————————————————————————

data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(
    ['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
print(df.head())

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
s = sns.lineplot(ax=ax1, x="Hour", y="Sum", data=df.loc[df.Class == 0])
s = sns.lineplot(ax=ax2, x="Hour", y="Sum", data=df.loc[df.Class == 1], color="red")
plt.suptitle("Total Amount")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
s = sns.lineplot(ax=ax1, x="Hour", y="Mean", data=df.loc[df.Class == 0])
s = sns.lineplot(ax=ax2, x="Hour", y="Mean", data=df.loc[df.Class == 1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
s = sns.lineplot(ax=ax1, x="Hour", y="Max", data=df.loc[df.Class == 0])
s = sns.lineplot(ax=ax2, x="Hour", y="Max", data=df.loc[df.Class == 1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
s = sns.lineplot(ax=ax1, x="Hour", y="Median", data=df.loc[df.Class == 0])
s = sns.lineplot(ax=ax2, x="Hour", y="Median", data=df.loc[df.Class == 1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
s = sns.lineplot(ax=ax1, x="Hour", y="Min", data=df.loc[df.Class == 0])
s = sns.lineplot(ax=ax2, x="Hour", y="Min", data=df.loc[df.Class == 1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
s = sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=data_df, palette="PRGn", showfliers=True)
s = sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=data_df, palette="PRGn", showfliers=False)
plt.show()
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————

tmp = data_df[['Amount', 'Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()

# ———————— features correlation ——————————————————————————————————————————————————————————————————————————————

plt.figure(figsize=(14, 14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="Reds")
plt.show()

# V1-V28之间没有显着的相关性。时间与V3呈反相关，数量与V7和V20成正相关，数量与V1和V5成反相关
# class=1 欺诈

# ———————— 把相关变量画图

# s = sns.lmplot(x='V20', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# s = sns.lmplot(x='V7', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# plt.show()
#
# # 确认数量和V7 V20 正相关的（Class = 0的回归线具有正斜率，而Class = 1的回归线具有较小的正斜率）。
#
# s = sns.lmplot(x='V2', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# s = sns.lmplot(x='V5', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
# plt.show()
#
# # 确认数量和V2 V5 负相关的（Class = 0的回归线具有正斜率，而Class = 1的回归线具有较小的正斜率）。

# 特征密度图
var = data_df.columns.values

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8, 4, figsize=(16, 28))

for feature in var:
    i += 1
    plt.subplot(8, 4, i)
    sns.kdeplot(t0[feature], bw=0.5, label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5, label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()

plt.close()

# V4，V11区别很大，V12，V14，V18有些区别，V1，V2，V3 ，V10部分区别，而V25，V26，V28表现类似。

# 除了时间和金额，合法交易的特征分布（Class = 0的值）以0为中心，有时在其中一个末端排队很长。而欺诈性交易（Class的值= 1）不对称分布。

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# —————— 预测模型

