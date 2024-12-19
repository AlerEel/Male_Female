import pandas as pd
import ast

train_data = pd.read_csv('data/train.csv',delimiter=';', header=0)
train_data = train_data.dropna(subset=["user_agent"])
train_data = train_data.reset_index()

agent_data = pd.DataFrame([ast.literal_eval(str(train_data["user_agent"][0]))], index=[0])

for i in range(1, len(train_data["user_agent"])-1):
    agent_dict = ast.literal_eval(str(train_data["user_agent"][i]))
    temp_df = pd.DataFrame.from_records([agent_dict])
    agent_data = pd.concat([agent_data, temp_df], ignore_index=True)
    print(i)

agent_data.to_csv('data/agent_data_train.csv', sep=';', index=False, encoding='utf-8')
