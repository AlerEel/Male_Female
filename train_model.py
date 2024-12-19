import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid

# Загрузка данных
train_labels = pd.read_csv('train_labels.csv',delimiter=';', header=0)
train_data = pd.read_csv('train.csv',delimiter=';', header=0)
test_data = pd.read_csv('test.csv',delimiter=';', header=0)
referer_vectors = pd.read_csv('referer_vectors.csv',delimiter=';', header=0)
geo_info = pd.read_csv('geo_info.csv',delimiter=';', header=0)
agent_data = pd.read_csv('agent_data_train.csv',delimiter=';', header=0)
print('Download success')

train_data = train_data.dropna(subset=["user_agent"])
train_data = train_data.reset_index()
train = pd.concat([train_data, agent_data], axis=1)

# Объединение train_data с train_labels
train = train.merge(train_labels, on='user_id', how='left')
# Объединение с referer_vectors
train = train.merge(referer_vectors, on='referer', how='left')
# Объединение с geo_info
#train = train.merge(geo_info, on='geo_id', how='left')
# Обработка пропущенных значений
#train.fillna(-1, inplace=True)
print('Merge success')

#rain['user_agent'] = train['user_agent'].astype('category').cat.codes
#train['timezone_id'] = train['timezone_id'].astype('category').cat.codes
#train['region_id'] = train['region_id'].astype('category').cat.codes
#train['country_id'] = train['country_id'].astype('category').cat.codes
#train['request_ts'] = train['request_ts'].astype('category').cat.codes
#train['referer'] = train['referer'].astype('category').cat.codes
#train['component0'] = train['component0'].astype('category').cat.codes
#train['component1'] = train['component1'].astype('category').cat.codes
#train['component2'] = train['component2'].astype('category').cat.codes
#train['component3'] = train['component3'].astype('category').cat.codes
#train['component4'] = train['component4'].astype('category').cat.codes
#train['component5'] = train['component5'].astype('category').cat.codes
#train['component6'] = train['component6'].astype('category').cat.codes
#train['component7'] = train['component7'].astype('category').cat.codes
#train['component8'] = train['component8'].astype('category').cat.codes
#train['component9'] = train['component9'].astype('category').cat.codes
#train['browser'] = train['browser'].astype('category').cat.codes
#train['os'] = train['os'].astype('category').cat.codes
#train['browser_version'] = train['browser_version'].astype('category').cat.codes
#train['os_version'] = train['os_version'].astype('category').cat.codes
print('Category success')
train = train.reindex(columns=['user_id', 'request_ts', 'referer', 'component0', 'component1',
                                     'component2', 'component3', 'component4', 'component5',
                                     'component6', 'component7', 'component8', 'component9','target'])
train = train.dropna(subset=["target"])
train = train.reset_index()
# Разделение данных на признаки и целевую переменную
# X = train.drop(columns=['user_id', 'target', 'request_ts', 'referer'])
X_Cat = train.drop(columns=['user_id','target','referer'])
y_Cat = train['target']

# Разделение на обучающую и валидационную выборки
X_train, X_test, y_train, y_test = train_test_split(X_Cat, y_Cat, test_size=0.2, random_state=42)

# Определение параметров для поиска
param_grid = {
    'iterations': [900, 1000, 1100, 1200],
    'depth': [8, 9, 10],
   'learning_rate': [0.1, 0.2, 0.3],
    'l2_leaf_reg': [5, 7, 9]
}

print('Fit start')

# Создание сетки параметров
grid = list(ParameterGrid(param_grid))

# Переменные для хранения результатов
results = []
total_combinations = len(grid)

# Создание директории для сохранения моделей
os.makedirs('saved_models', exist_ok=True)

# Перебор всех комбинаций параметров
for i, params in enumerate(grid):
    print(f"Running combination {i + 1}/{total_combinations}: {params}")

    # Создание и обучение модели
    model = CatBoostClassifier(**params, silent=True)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Сохранение результатов
    results.append({'params': params, 'accuracy': accuracy})
    print(f"Accuracy: {accuracy:.4f}")

    # Сохранение модели
    model_filename = f"saved_models/model_{i + 1}_acc_{accuracy:.4f}.cbm"
    model.save_model(model_filename)
    print(f"Model saved as: {model_filename}")

# Поиск лучших параметров
best_result = max(results, key=lambda x: x['accuracy'])
print(f"Best parameters: {best_result['params']}")
print(f"Best accuracy: {best_result['accuracy']:.4f}")

# Запись результатов в файл
print(results)
f = open('results/result.txt', 'w')  # открытие в режиме записи
f.write(json.dumps(results))
f.close()
f = open('results/best_result.txt', 'w')  # открытие в режиме записи
f.write(json.dumps(best_result))
f.close()