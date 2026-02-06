import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('house.csv')

y=data.Price
features=data.drop(['Price'],axis=1)
x=features.select_dtypes(exclude=['object','bool'])
train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

missing_col=[col for col in train_x.columns if data[col].isnull().any()]
reduced_train_x=train_x.drop(missing_col,axis=1)
reduced_val_x=val_x.drop(missing_col,axis=1)

model=RandomForestRegressor(n_estimators=10,random_state=0)
model.fit(reduced_train_x,train_y)
predict_x=model.predict(reduced_val_x)
error=mean_absolute_error(val_y,predict_x)
print(error)