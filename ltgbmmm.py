import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

RANDOM_STATE = 2018
print('Loading data...')
# 加载或构造数据

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

# estimator = lgb.LGBMRegressor(num_leaves=31)  0.1  40

#
# estimator = LogisticRegression(solver='liblinear')
# param_grid = {
#     'penalty': ['l1', 'l2'],
#     'class_weight': ['balanced', None],
#     'C': [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# }
#
# CV_lr = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='recall', verbose=1, n_jobs=-1)
# CV_lr.fit(X_train, y_train)
#
# best_parameters = CV_lr.best_params_
# print('The best parameters for using this model is', best_parameters)
# # {'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l2'}
# # {'C': 100, 'class_weight': 'balanced', 'penalty': 'l1'}
# # {'C': 60, 'class_weight': 'balanced', 'penalty': 'l1'}
#
# estimator = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                                    max_depth=None, max_features='auto', max_leaf_nodes=None,
#                                    min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
#                                    random_state=None)  # gridsearchcv()中的分类器
# param = {
#     "n_estimators": [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
#     "criterion": ["gini", "entropy"],
#     "min_samples_leaf": [2, 4, 6, 8, 10, 12],
# }
# clf = GridSearchCV(estimator=estimator, param_grid=param, cv=5)  # 网格搜索来调参
# clf.fit(X_train, y_train)
#
# best_param = clf.best_params_  # 最优的参数，类型为字典dict
# print('The best parameters for using this model is', best_param)
# # {'criterion': 'gini', 'min_samples_leaf': 2, 'n_estimators': 10}
# # {'criterion': 'entropy', 'min_samples_leaf': 4, 'n_estimators': 60}
# # {'criterion': 'entropy', 'min_samples_leaf': 2, 'n_estimators': 30}
#
# estimator = AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME.R')
#
# # 网格搜索，参数优化
# # param_grid = {
# #     'learning_rate': [0.01, 0.1, 1],
# #     'n_estimators': [20, 40]
# # }
#
# # param_grid = {"base_estimator__criterion": ["gini", "entropy"],
# #               "base_estimator__splitter": ["best", "random"],
# #               'learning_rate': [0.01, 0.1, 0.5, 1],
# #               'n_estimators': [5, 10, 20, 30, 40]
# #               }
#
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
#     'n_estimators': [50, 60, 70, 80, 90, 100, 200]
# }
#
# ccc = GridSearchCV(estimator, param_grid, cv=3)
# ccc.fit(X_train, y_train)
#
# print('Best parameters found by grid search are:', ccc.best_params_)
# # {'learning_rate': 0.1, 'n_estimators': 90}
# # {'learning_rate': 0.5, 'n_estimators': 100}【】【】【】【】【】
#
#
# # The best parameters for using this model is {'C': 110, 'class_weight': 'balanced', 'penalty': 'l1'}
# # The best parameters for using this model is {'criterion': 'gini', 'min_samples_leaf': 6, 'n_estimators': 70}
# # Best parameters found by grid search are: {'learning_rate': 0.1, 'n_estimators': 50}
#
#
# # The best parameters for using this model is {'C': 70, 'class_weight': 'balanced', 'penalty': 'l1'}
# # The best parameters for using this model is {'criterion': 'gini', 'min_samples_leaf': 2, 'n_estimators': 10}
# # Best parameters found by grid search are: {'learning_rate': 1, 'n_estimators': 200}
#

estimator = CatBoostClassifier(iterations=500,
                               eval_metric='AUC',
                               random_seed=RANDOM_STATE,
                               bagging_temperature=0.2,
                               od_type='Iter',
                               metric_period=50,
                               od_wait=100, learning_rate=0.1, l2_leaf_reg=1,depth=6)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [500], 'depth': [4, 6, 8], 'l2_leaf_reg': [1, 4, 9]
}

ccc = GridSearchCV(estimator, param_grid)
ccc.fit(X_train, y_train)

print('Best parameters found by grid search are:', ccc.best_params_)

# Best parameters found by grid search are: {'depth': 6, 'iterations': 500, 'l2_leaf_reg': 1, 'learning_rate': 0.1}


