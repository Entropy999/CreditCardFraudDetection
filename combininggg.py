import pandas as pd
import numpy as np
from matplotlib import gridspec
import itertools
# import cufflinks as cf
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
from lightgbm import LGBMClassifier
# import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE, ADASYN
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

# ———————————————————————— 选择是否降采样 ————————————————————————————

# X = data_df.drop(['Class'], axis=1)
# y = data_df['Class']

X = data_df_under.drop(['Class'], axis=1)
y = data_df_under['Class']
print(sorted(Counter(y).items()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


# ———————————————————————— 选择是否降采样 ————————————————————————————

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix"',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_metrics():
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]
    print('Precision = %.03f' % (tp / (tp + fp)))
    print('Recall    = %.03f' % (tp / (tp + fn)))
    print('F1_score  = %.03f' % (2 * (((tp / (tp + fp)) * (tp / (tp + fn))) /
                                      ((tp / (tp + fp)) + (tp / (tp + fn))))))


def plot_precision_recall():
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.show()


def plot_roc():
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc, linewidth=2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 0.001])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


class_names = [0, 1]
target_names = ['class 0', 'class 1']

# —————————— 逻辑回归
print('LogisticRegression')
clf = LogisticRegression(C=60, penalty='l1', solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
show_metrics()
print(classification_report(y_test, y_pred, target_names=target_names))
fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()

print('rocauc=')
print(roc_auc_score(y_test, y_pred))
fpr_lr, tpr_lr, t_lr = fpr, tpr, t
precision_lr, recall_lr, thresholds_lr = precision, recall, thresholds

# —————————— 随机森林
print('RandomForestClassifier')
clf = RandomForestClassifier(n_jobs=NO_JOBS,
                             random_state=RANDOM_STATE, min_samples_leaf=2,
                             criterion='entropy',
                             n_estimators=30,
                             verbose=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
show_metrics()
print(classification_report(y_test, y_pred, target_names=target_names))
fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()

print('rocauc=')
print(roc_auc_score(y_test, y_pred))
fpr_rfc, tpr_rfc, t_rfc = fpr, tpr, t
precision_rfc, recall_rfc, thresholds_rfc = precision, recall, thresholds

# —————————— XGB

# print('xgb')
# clf = xgb.XGBClassifier(n_jobs=-1)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_score = clf.predict_proba(X_test)[:, 1]
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# show_metrics()
# print(classification_report(y_test, y_pred, target_names=target_names))
# fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()
#
# fpr_xgb, tpr_xgb, t_xgb = fpr, tpr, t
# precision_xgb, recall_xgb, thresholds_xgb = precision, recall, thresholds

# —————————— cat
print('CatBoostClassifier')
clf = CatBoostClassifier(iterations=500,
                         learning_rate=0.1,
                         depth=6,
                         eval_metric='AUC',
                         random_seed=RANDOM_STATE,
                         bagging_temperature=0.2,
                         od_type='Iter',
                         metric_period=VERBOSE_EVAL,
                         od_wait=100, l2_leaf_reg=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
show_metrics()
print(classification_report(y_test, y_pred, target_names=target_names))
fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()

print('rocauc=')
print(roc_auc_score(y_test, y_pred))
fpr_cat, tpr_cat, t_cat = fpr, tpr, t
precision_cat, recall_cat, thresholds_cat = precision, recall, thresholds

# —————————— AdaBoostClassifier
print('AdaBoostClassifier')
clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.5,
                         n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
show_metrics()
print(classification_report(y_test, y_pred, target_names=target_names))
fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()

print('rocauc=')
print(roc_auc_score(y_test, y_pred))
fpr_ada, tpr_ada, t_ada = fpr, tpr, t
precision_ada, recall_ada, thresholds_ada = precision, recall, thresholds

# —————————— lightgbm
print('lightgbm')
clf = LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
show_metrics()
print(classification_report(y_test, y_pred, target_names=target_names))
fpr, tpr, t = roc_curve(y_test, y_score)
# plot_roc()
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# plot_precision_recall()
print('rocauc=')
print(roc_auc_score(y_test, y_pred))


fpr_lgbm, tpr_lgbm, t_lgbm = fpr, tpr, t
precision_lgbm, recall_lgbm, thresholds_lgbm = precision, recall, thresholds


# ———————————————— Precision Recall Curve 合集

# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# cf.set_config_file(offline=True)

def prec_recall_all_models():
    plt.figure(figsize=(16, 12))
    plt.plot(recall_lr, precision_lr, linewidth=2)
    plt.plot(recall_rfc, precision_rfc, linewidth=2)
    # plt.plot(recall_xgb, precision_xgb, linewidth=2)
    plt.plot(recall_cat, precision_cat, linewidth=2)
    plt.plot(recall_ada, precision_ada, linewidth=2)
    plt.plot(recall_lgbm, precision_lgbm, linewidth=2)

    plt.scatter(rec, pr, linewidth=2, color='red')
    plt.axvline(rec, color='red', linewidth=1, linestyle='--')
    plt.axhline(pr, color='red', linewidth=1, linestyle='--')
    plt.xlim([0.5, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve - PR = 0.878 - REC = 0.878 - F1 = 0.878 ')
    plt.legend(['LogisticRegression', 'RandomForest', 'CatBoost', 'AdaBoost', 'lightGBM'], loc='lower right')
    plt.savefig('prc1.png')
    plt.show()


rec = 0.878
pr = 0.878
prec_recall_all_models()


def roc_curve_all_models():
    plt.figure(figsize=(16, 12))
    plt.plot(fpr_lr, tpr_lr, label='LogisticRegression ROC Curve', linewidth=2)
    plt.plot(fpr_rfc, tpr_rfc, label='RandomForest ROC Curve', linewidth=2)
    plt.plot(fpr_cat, tpr_cat, label='CatBoost ROC Curve', linewidth=2)
    plt.plot(fpr_ada, tpr_ada, label='AdaBoost ROC Curve', linewidth=2)
    plt.plot(fpr_lgbm, tpr_lgbm, label='lightGBM ROC Curve', linewidth=2)

    plt.plot(recall_lr, precision_lr, linewidth=2)
    plt.plot(recall_rfc, precision_rfc, linewidth=2)
    # plt.plot(recall_xgb, precision_xgb, linewidth=2)
    plt.plot(recall_cat, precision_cat, linewidth=2)
    plt.plot(recall_ada, precision_ada, linewidth=2)
    plt.plot(recall_lgbm, precision_lgbm, linewidth=2)


    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 0.8])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # plt.legend(['Rf', 'Xgb', 'Lr', 'Voting'], loc='upper left')
    plt.legend(['LogisticRegression', 'RandomForest', 'CatBoost', 'AdaBoost', 'lightGBM'], loc='lower right')
    plt.savefig('roc1.png')

    plt.show()

roc_curve_all_models ()


