import pandas as pd

data=pd.read_csv('demo.csv')

#print(data.describe())
#print(data.describe(include='all')) #include non numerical values

#print(data.GPA.mean())
#print(data.Age.unique())

print(data.Age.value_counts())# print the value as index and its count as value 

print(data.Age.value_counts(normalize=True))# print the value as percentage