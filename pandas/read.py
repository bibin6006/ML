import pandas as pd

f=pd.read_csv('demo.csv',index_col=0)
print(f.head()) #print first 5 rows
#print(f.shape)# print the shape (number of rows,number of column)
#print(f['Age']) #print age column

#print(f.loc[1])# index_col=0 means we set 0th column as a index  
#print(f.iloc[0])
#print(f.head())

# indexing ,selecting  
#iloc
#print(f.Age)# print age column 
#print(f.iloc[:,0]) #print all row's first column
#print(f.iloc[[0,1,2],0]) #print first 3 row's first column

#loc

#print(f.loc[1,'FirstName'])
#print(f)