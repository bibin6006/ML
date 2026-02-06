import pandas as pd

data={'id':[1,2,None],'std':['bob','alice',None]}

df=pd.DataFrame(data)
#print(df)
print(df.isnull()) # return true is value is None 

print(df.notnull())# retuen true is value is not None


