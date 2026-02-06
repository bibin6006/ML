import pandas as pd
data=pd.read_csv('multi_group.csv')
multi=data.groupby(['store','product']).sales.sum()
print(multi.loc[('A','Mug')])
idx=multi.reset_index()# reset_index() create index corresponding to rows
print(idx.loc[0])