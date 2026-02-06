import pandas as pd
f=pd.read_csv('demo.csv')
neww=f.set_index('StudentID')# it set studentid is an index 
print(neww.head())