import pandas as pd
data=pd.read_csv('demo.csv')
age=data.Age==20
#print(age)# print a series of true and false 
#print(data.loc[age])

gpa=data.GPA>3.5
#print(gpa)

#print(data.loc[age&gpa])

#print(data.loc[age|gpa])
#con=data.Age.isin([20,21])
#print(data.loc[con])
#print(con)
#print(data.loc[data.Age.isin([20,21])])

# add new column

data['neww']=range(1,len(data)+1)

print(data['neww'])

print(data)