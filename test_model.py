from catboost import CatBoostClassifier, Pool
import pandas as pd

# Загрузка модели из файла
model = CatBoostClassifier() # Или CatBoostRegressor()
model.load_model('saved_models/model_88_acc_0.7787.cbm')

# Загрузка тестовых данных
test_users = pd.read_csv('data/test_users.csv',delimiter=';', header=0)
test_data = pd.read_csv('data/test.csv',delimiter=';', header=0)
referer_vectors = pd.read_csv('data/referer_vectors.csv',delimiter=';', header=0)
test = test_users.merge(test_data, on='user_id', how='left')
test = test.merge(referer_vectors, on='referer', how='left')
test = test.reset_index()
feature_names = ['index','request_ts','component0', 'component1',
                                     'component2', 'component3', 'component4', 'component5',
                                     'component6', 'component7', 'component8', 'component9']

# Создание объекта Pool с указанием признаков
test_pool = Pool(data=test[feature_names])

# Получение предсказаний
predictions = model.predict(test_pool)

# Добавление предсказаний в DataFrame с тестовыми данными
test['target'] = predictions
result = test.reindex(columns=['user_id', 'target'])

# Вывод обновленного DataFrame
print("Обновленный DataFrame с предсказаниями:")
print(result)
result.to_csv('data/result.csv', sep=';', index=False, encoding='utf-8')