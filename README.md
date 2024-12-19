# Задача: Предскажи пол

Описание датасета, расположенного в папке data:\
Таблицы

train_labels.csv:
- «user_id» - id пользователя;
- «target» - пол пользователя (1 / 0).
 
train.csv, test.csv;
- «request_ts» - server timestamp of request;
- «user_id» - id пользователя (см. п.1);
- «referer» - url, где показывается реклама. В данном случае захэшировано 2 части url:
1) domain - домен урла;
2) path - все что после domain. Например, https://a758bf6/1432d3f1, a 758bf6 - domain, 1432d3f1 - path.
- «geo_id» - id geo;
- «user_agent» - строка user_agent.
 
referer_vectors.csv:
- «component0» - … - «component9» - числа, которые несут в себе информацию о url. Их нельзя как-либо интерпретировать;
- «referer» - url, где показывается реклама (см. п.2).
 
geo_info.csv:
- «geo_id» - id geo (см. п.2);
- «country_id» - id страны;
- «region_id» - id региона;
- «timezone» - часовой пояс для geo.

agent_data_train.csv(преобразованные в таблицу столбец "user_agent" из train.csv:
- «browser» - Браузер пользователя;
- «browser_version» - Версия браузера;
- «os» - ОС пользователя;
- «os_version» - Версия ОС.

agent_data_test.csv(преобразованные в таблицу столбец "user_agent" из test.csv:
- «browser» - Браузер пользователя;
- «browser_version» - Версия браузера;
- «os» - ОС пользователя;
- «os_version» - Версия ОС.

## Требования
Для каждого пользователя (user) из файла test_users.csv необходимо предсказать пол. Их рекламные запросы лежат в файле test.csv.

Формат вывода
Формат вывода соответствует train_labels.csv.

## Команда:

Алексей Таланов\
Алексей Касаткин\
Павел Соколов

## Структура проекта:
catboost_info - папка с данными по ходу работы модели CatBoost\
data - папка с данными, которые предоставлены заказчиком(описаны выше)\
results - папка с результатами
saved_models - папка с моделями, из которых выбирается с лучшими результатами

## Результаты:
Нами получена точность 82%\
Были проверены различные методы обработки данных и модели для классификации(ноутбук по ссылке)\
-https://colab.research.google.com/drive/1-5Pc5mQFE8IUZwRBnD1hQ7c5Bl3nG18V#scrollTo=lFOd-_LLkfV1
Наилучшие результаты показала модель CatBoost\

### Программное окружение:
Python - 3.12(на версию 3.13 CatBoost на установился)\
Используемые библиотеки в файле requirements.txt
