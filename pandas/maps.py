import pandas as pd
data=pd.read_csv('demo.csv')
#print(data)
mean=data.GPA.mean()
print(mean)
# map()
#map=data.GPA.map(lambda x:x-mean)
#def find(val):
#    return val-mean
#map=data.GPA.map(find)    
#print(map)

#apply()
def find(row):
    print(row)
    return row.GPA-mean
d=data.apply(find,axis='columns')
print(d)
