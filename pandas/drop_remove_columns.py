import pandas as pd
data=pd.read_csv('demo.csv')


x=data.select_dtypes(exclude=['object','bool']) #this line of code remove the columns with object,and boolean and create new dataframe assign to x 
new_x=x.drop(['Age'],axis=1) # it remove the Age column 
print(new_x)