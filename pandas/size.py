import pandas as pd
data=pd.read_csv('size.csv')
# print(data)
print(data.groupby('group').count())
print(data.groupby('group').size())