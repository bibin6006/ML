import pandas as pd

data=[1,2,3,4]
series=pd.Series(data,index=['one','two','three','four'])

#print(series)
#print(series['two'])
#print(series.loc['four'])
#print(series.iloc[0])
print(series[series>2])
