import numpy as np
import pandas as pd
# import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from collections import Counter
import xgboost as xgb

print('Loading data...')
# 加载或构造数据

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

# ———————————————————————— 选择是否降采样 ————————————————————————————

# X = data_df.drop(['Class'], axis=1)
# y = data_df['Class']

X = data_df_under.drop(['Class'], axis=1)
y = data_df_under['Class']
print(sorted(Counter(y).items()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

# print('Starting training...')
# # 模型训练
# gbm = lgb.LGBMRegressor(num_leaves=31,
#                         learning_rate=0.05,
#                         n_estimators=20)
# gbm.fit(X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric='l1',
#         early_stopping_rounds=5)
#
# print('Starting predicting...')
# # 模型预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# # 模型验证
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
#
# # 特征重要性
# print('Feature importances:', list(gbm.feature_importances_))
#
#
# # 自定义eval_metric
# # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# # Root Mean Squared Logarithmic Error (RMSLE)
# def rmsle(y_true, y_pred):
#     return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False
#
#
# print('Starting training with custom eval function...')
# # 模型训练
# gbm.fit(X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric=rmsle,
#         early_stopping_rounds=5)
#
#
# # 自定义eval_metric
# # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# # Relative Absolute Error (RAE)
# def rae(y_true, y_pred):
#     return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False
#
#
# print('Starting training with multiple custom eval functions...')
# # 模型训练
# gbm.fit(X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric=[rmsle, rae],
#         early_stopping_rounds=5)
#
# print('Starting predicting...')
# # 模型预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# # 模型评估
# print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])
# print('The rae of prediction is:', rae(y_test, y_pred)[1])
#
# # other scikit-learn modules
# estimator = lgb.LGBMRegressor(num_leaves=31)
#
# # 网格搜索，参数优化
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [20, 40]
# }
# gbm = GridSearchCV(estimator, param_grid, cv=3)
# gbm.fit(X_train, y_train)
#
# print('Best parameters found by grid search are:', gbm.best_params_)
#


# print(range(80, 200, 4))
#
# print(np.linspace(0.5, 0.98, 10))


estimator = xgb.XGBClassifier(n_jobs=-1)
# param_grid = {
#     'n_estimators': [20, 40, 80, 100, 200],
#     'max_depth': [2, 4, 6, 8, 10, 12],
#     'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
#     'subsample': np.linspace(0.7, 0.9, 3),
#     'colsample_bytree': np.linspace(0.5, 0.98, 3),
#     'min_child_weight': [1, 3, 5]
# }
param_grid = {
    'n_estimators': [120],
    'learning_rate': [0.1], 'max_depth': [2, 4, 6], 'subsample': np.linspace(0.1, 0.6, 3),
    'colsample_bytree': np.linspace(0.1, 0.5, 2),
    'min_child_weight': [1, 3, 5]
}
ccc = GridSearchCV(estimator, param_grid)
ccc.fit(X_train, y_train)

print('Best parameters found by grid search are:', ccc.best_params_)
# Best parameters found by grid search are: {'learning_rate': 0.1, 'n_estimators': 120}
#  {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 120, 'subsample': 0.7}

clf = xgb.XGBClassifier(colsample_bytree=0.5,
                         learning_rate=0.1,
                         max_depth=2,
                         min_child_weight=5,
                         n_estimators=120,
                         subsample=0.35)
# 就这个啦
# {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 5, 'n_estimators': 120, 'subsample': 0.35}