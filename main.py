import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import ast
from sklearn.model_selection import GridSearchCV

# Загрузка данных
train_labels = pd.read_csv('train_labels.csv',delimiter=';', header=0)
train_data = pd.read_csv('train.csv',delimiter=';', header=0)
test_data = pd.read_csv('test.csv',delimiter=';', header=0)
referer_vectors = pd.read_csv('referer_vectors.csv',delimiter=';', header=0)
geo_info = pd.read_csv('geo_info.csv',delimiter=';', header=0)
agent_data = pd.read_csv('agent_data.csv',delimiter=';', header=0)
print('Download success')

train_data = train_data.dropna(subset=["user_agent"])
train_data = train_data.reset_index()
train = pd.concat([train_data, agent_data], axis=1)

# Объединение train_data с train_labels
train = train.merge(train_labels, on='user_id')
# Объединение с referer_vectors
train = train.merge(referer_vectors, on='referer', how='left')
# Объединение с geo_info
train = train.merge(geo_info, on='geo_id', how='left')
# Обработка пропущенных значений
train.fillna(-1, inplace=True)
print('Merge success')

train['user_agent'] = train['user_agent'].astype('category').cat.codes
train['timezone_id'] = train['timezone_id'].astype('category').cat.codes
train['region_id'] = train['region_id'].astype('category').cat.codes
train['country_id'] = train['country_id'].astype('category').cat.codes
train['request_ts'] = train['request_ts'].astype('category').cat.codes
train['referer'] = train['referer'].astype('category').cat.codes
train['component0'] = train['component0'].astype('category').cat.codes
train['component1'] = train['component1'].astype('category').cat.codes
train['component2'] = train['component2'].astype('category').cat.codes
train['component3'] = train['component3'].astype('category').cat.codes
train['component4'] = train['component4'].astype('category').cat.codes
train['component5'] = train['component5'].astype('category').cat.codes
train['component6'] = train['component6'].astype('category').cat.codes
train['component7'] = train['component7'].astype('category').cat.codes
train['component8'] = train['component8'].astype('category').cat.codes
train['component9'] = train['component9'].astype('category').cat.codes
train['browser'] = train['browser'].astype('category').cat.codes
train['os'] = train['os'].astype('category').cat.codes
train['browser_version'] = train['browser_version'].astype('category').cat.codes
train['os_version'] = train['os_version'].astype('category').cat.codes
print('Category success')

# Разделение данных на признаки и целевую переменную
# X = train.drop(columns=['user_id', 'target', 'request_ts', 'referer'])
X_Cat = train.drop(columns=['user_id','target','referer', 'user_agent'])
y_Cat = train['target']

# Разделение на обучающую и валидационную выборки
X_train_Cat, X_val_Cat, y_train_Cat, y_val_Cat = train_test_split(X_Cat, y_Cat, test_size=0.2, random_state=42)

# Определение параметров для поиска
param_grid = {
    'iterations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 100],
    'depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
print('Fit start')
# Создание и обучение модели с автоматической обработкой категориальных данных
model_Cat = CatBoostClassifier(silent=True)
grid_search = GridSearchCV(estimator=model_Cat, param_grid=param_grid, cv=3, scoring='accuracy')
# Обучение модели с подбором гиперпараметров
grid_search.fit(X_train_Cat, y_train_Cat)

# Лучшие параметры
print(f'Best parameters: {grid_search.best_params_}')
f = open('xyz.txt','w')  # открытие в режиме записи
f.write(grid_search.best_params_)
f.close()