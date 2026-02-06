import pandas as pd

df=pd.DataFrame({'student':['bibin','elbin','anu'],'admission No':[1,2,3]},index=['std1','std2','std3'])
#print(df)

#print(df['student']) # it print the student column

#print(df.loc['std1']) # to select row 

#print(df.iloc[0])# select row by indexing 

#print(df.iloc[0,0]) #the first value of row and second value of column

#other way to define dataframe

#data={'student':['bibin','elbin','anu'],'admission No':[1,2,3]}

#print(pd.DataFrame(data,index=['std1','std2','std3']))


"""data=[{'student':'bibin','admission NO':1},
      {'student':'elbin','admission NO':2},
      {'student':'anu','admission NO':3}]

print(pd.DataFrame(data))
"""
data=[['bibin',1],
      ['elbin',2],
      ['anu',3]]
# df=pd.DataFrame(data,columns=['student','admission No'])
# print(df)
# df.iloc[0]=['athul',99]
print(df)

