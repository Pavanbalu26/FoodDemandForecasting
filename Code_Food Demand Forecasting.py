import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor, plot_importance
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score




#Data Analysis
#Analysing MeaL_Info

meal_info = pd.read_csv("meal_info.csv")
meal_info.head()
meal_info.info()
meal_info.category.value_counts()

meal_info.cuisine.value_counts()


#Analysing Fulfilment¬_Center_Info

fulfilment_center_info = pd.read_csv("fulfilment_center_info.csv") 
fulfilment_center_info.head()
fulfilment_center_info.city_code.value_counts()

fulfilment_center_info.info()
fulfilment_center_info.region_code.value_counts()
fulfilment_center_info.center_type.value_counts()
fulfilment_center_info.op_area.describe()

#Analysing Train¬ Data

train_data = pd.read_csv("train.csv")
train_data.head()
train_data.info()
train_data.week.value_counts()
train_data.center_id.value_counts()

train_data.meal_id.value_counts()
train_data.checkout_price.describe()

train_data.base_price.describe()

train_data.sort_values('checkout_price').head()
train_data.num_orders.describe()
train_data.sort_values('num_orders').tail()
train_data.emailer_for_promotion.value_counts()
train_data.homepage_featured.value_counts()

#Analysing test data

test_data = pd.read_csv("test.csv")
test_data.head()
test_data.info()
test_data.week.value_counts()
test_data.meal_id.value_counts()
test_data.center_id.value_counts()
test_data.checkout_price.describe()
test_data.base_price.describe()
test_data.emailer_for_promotion.value_counts()
test_data.homepage_featured.value_counts()
sns.lineplot(train['checkout_price'],train['num_orders'])

#Visualisation
sns.lineplot(train_data['checkout_price'],train_data['num_orders'])
sns.boxplot(x='num_orders',data=train_data)
sns.jointplot(x='checkout_price',y='num_orders',data=train_data)

#Data Preprocessing
#Combining  train and test data
train_data_without_target = train_data[train_data.columns[train_data.columns != 'num_orders'].values]
total_data = train_data_without_target.append(test_data, sort=False)
total_data.tail()


#Merge dataset
total_data = total_data.merge(fulfilment_center_info, on='center_id', how='left') 
total_data.head()
total_data = total_data.merge(meal_info, on='meal_id', how='left') 
total_data.head()
total_data.info()

#Derive new variable
meal_base_price = total_data[['week', 'center_id', 'meal_id', 'base_price']]
meal_base_price = meal_base_price.set_index(['meal_id', 'center_id', 'week'])
meal_base_price = meal_base_price.sort_index()
meal_base_price.head()
meal_per_center = train_data[['week', 'center_id', 'meal_id', 'num_orders']]
meal_per_center = meal_per_center.set_index(['meal_id', 'center_id', 'week'])
meal_per_center = meal_per_center.sort_index()
meal_per_center.head()
meal_across_center = train_data[['week', 'meal_id', 'num_orders']] 
meal_across_center = meal_across_center.set_index(['meal_id', 'week']) 
meal_across_center = meal_across_center.sort_index() 
meal_across_center.head()


def average_orders(row):
    if (row.meal_id, row.center_id) in meal_base_price.index:
        row['mean_base_price'] = meal_base_price.loc[(row.meal_id, row.center_id)].loc[:row.week].base_price.mean()
    else:
        row['mean_base_price'] = row['base_price']
        
    week_adj = row.week - 10
    if (row.meal_id, row.center_id) in meal_per_center.index:
        history = meal_per_center.loc[(row.meal_id, row.center_id)]
        row['average_orders_13week'] = history.loc[(row.week-13):row.week].num_orders.mean()
        row['average_orders_26week'] = history.loc[(row.week-26):row.week].num_orders.mean()
        row['average_orders_52week'] = history.loc[(row.week-52):row.week].num_orders.mean()
        row['average_orders_13week_adj'] = history.loc[(week_adj-13):week_adj].num_orders.mean()
        row['average_orders_26week_adj'] = history.loc[(week_adj-26):week_adj].num_orders.mean()
    else:
        row['average_orders_13week'] = 0
        row['average_orders_26week'] = 0
        row['average_orders_52week'] = 0
        row['average_orders_13week_adj'] = 0
        row['average_orders_26week_adj'] = 0
        
    if row.meal_id in meal_across_center.index:
        history_across = meal_across_center.loc[row.meal_id]
        row['average_orders_13week_across'] = history_across.loc[(row.week-13):row.week].num_orders.mean()
        row['average_orders_26week_across'] = history_across.loc[(row.week-26):row.week].num_orders.mean()
        row['average_orders_52week_across'] = history_across.loc[(row.week-52):row.week].num_orders.mean()
        row['average_orders_13week_adj_across'] = history_across.loc[(week_adj-13):week_adj].num_orders.mean()
        row['average_orders_26week_adj_across'] = history_across.loc[(week_adj-26):week_adj].num_orders.mean()
    else:
        row['average_orders_13week_across'] = 0
        row['average_orders_26week_across'] = 0
        row['average_orders_52week_across'] = 0
        row['average_orders_13week_adj_across'] = 0
        row['average_orders_26week_adj_across'] = 0
    
    return row
    
total_data = total_data.apply(average_orders, axis=1)
total_data['average_orders_13week'].describe()
total_data['discount'] = total_data['mean_base_price'] - total_data['checkout_price'] 
total_data['discount'] = total_data['discount'] / total_data['mean_base_price'] 
total_data.discount.describe()
total_data['year'] = (((total_data['week'] - 1)/52) + 1).astype('int')
total_data.year.value_counts()
total_data['month'] = (((total_data['week'] - 1)/4).astype('int') % 13) + 1
total_data.month.value_counts()
total_data['quarter'] = (((total_data['week'] - 1)/13).astype('int') % 4) + 1
total_data.quarter.value_counts()
total_data['week_in_month'] = (((total_data['week'] - 1) % 4) + 1) 
total_data.week_in_month.value_counts()



#Splitting total data into train and test
train_data = train_data[['id', 'num_orders']].merge(total_data, on='id', how='left')
train_data.head()
training_data.info()
train_data.to_csv('train_feature.csv', index=False)
test_data = test_data[['id']].merge(total_data, on='id', how='left')
test_data.head()
test_data.info()
test_data.to_csv('test_feature.csv', index=False)

#Defining constants
target = 'num_orders'

features = ['center_id', 'meal_id', 'checkout_price', 'mean_base_price', 'discount', 'emailer_for_promotion', 
            'homepage_featured', 'city_code', 'center_type', 'category', 'year', 'region_code', 'month', 
            'week_in_month', 'cuisine', 'average_orders_26week_adj', 'average_orders_52week', 
            'average_orders_26week', 'average_orders_26week_adj_across', 'average_orders_26week_across']

categorical_columns = ['week', 'center_id', 'meal_id', 'emailer_for_promotion', 'homepage_featured', 
                       'city_code', 'region_code', 'center_type', 'category', 'cuisine', 'year', 'month', 'quarter', 
                       'week_in_month']

encoded_columns = ['center_id_55', 'meal_id_1885', 'emailer_for_promotion_0', 'homepage_featured_0', 'city_code_647', 
                   'region_code_56', 'center_type_TYPE_C', 'category_Beverages', 'cuisine_Italian', 'year_3', 
                   'month_1', 'week_in_month_2']

#Reading Data
trainset = pd.read_csv("train_feature.csv", index_col='id').fillna(0)
trainset.head()
trainset.cuisine.value_counts()
testset = pd.read_csv("test_feature.csv", index_col='id').fillna(0)
testset.head()

testset.info()

#Data Preprocessing
def preprocess(trainset, testset, remove_outliers=False):
    
    if remove_outliers:
        trainset = trainset[trainset.num_orders <= 20000]
        trainset = trainset[trainset.checkout_price >= 3]
        
    dataset = trainset.append(testset, sort=False).fillna(0)
    
    for column in categorical_columns:
        dataset[column] = dataset[column].astype('category')
    
    dataset = dataset[features]
    
    dataset = pd.get_dummies(dataset[features])
    dataset = dataset.drop(encoded_columns, axis=1)
    
    trainset = trainset[[target]].join(dataset)
    testset = testset[[]].join(dataset)
    
    return trainset, testset


trainset, testset = preprocess(trainset, testset) 
print("Trainset size: {}".format(trainset.shape))
print("Testset size: {}".format(testset.shape))
feature_columns = trainset.columns
feature_columns = feature_columns[feature_columns != target]
feature_columns.shape
trainset, validationset = train_test_split(trainset, random_state=41, test_size=0.2)
X_train, y_train = trainset[feature_columns], np.log(trainset[target])
X_val, y_val = validationset[feature_columns], np.log(validationset[target])

print("Train set size: {}".format(X_train.shape))
print("Validation set size: {}".format(X_val.shape))

#6.2Training Models
#Decision tree regression
DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))
score_DT = r2_score(y_pred,y_val)
score_DT

#KNeighborsRegression
KNN = KNeighborsRegressor()
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))
score_KNN = r2_score(y_pred,y_val)
score_KNN
#RandomForestRegression
RFR=RandomForestRegressor()
RFR.fit(X_train, y_train)
y_pred = RFR.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))
score_RFR = r2_score(y_pred,y_val)
score_RFR

#AdaBoostRegressor

ABR=AdaBoostRegressor()
ABR.fit(X_train, y_train)
y_pred = ABR.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))
score_ABR = r2_score(y_pred,y_val)
score_ABR

#LightGBM

lgb = LGBMRegressor(importance_type='gain')
params = { 'num_leaves': [41, 51], 'n_estimators': [230, 260], 'min_child_samples': [40, 45, 50], 'random_state': [2019] }
lgb_grid = GridSearchCV(lgb, params, cv=5, scoring='neg_mean_squared_error', n_jobs=8) 
lgb_grid.fit(X_train, y_train)
lgb = lgb_grid.best_estimator_score =100*np.sqrt(metrics.mean_squared_log_error(y_val, lgb.predict(X_val)))
score_lgb = r2_score(y_val,lgb.predict(X_val))
score_lgb 
print('Best Estimator: {}'.format(lgb))
print('Best Score on validation: {}'.format(score))

ax = plot_importance(lgb_grid.best_estimator_, max_num_features=50, height=0.8, figsize=(12, 10))
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()

#XGBoost
xgb = XGBRegressor(objective='reg:squarederror', random_state=41,missing=0.0, n_jobs=8, 
                   max_depth=9, n_estimators=300, min_child_weight=45)
xgb.fit(X_train, y_train)
100*np.sqrt(metrics.mean_squared_log_error(y_val, xgb.predict(X_val)))
score_xgb = r2_score(y_val,xgb.predict(X_val))
score_xgb
#Submission
X_test = testset[feature_columns]
y_pred = (np.exp(lgb.predict(X_test)) + np.exp(xgb.predict(X_test))) / 2
testset[target] = np.round(y_pred, decimals=0)

submission = testset[[target]]
submission.to_csv('ensemble.csv')

