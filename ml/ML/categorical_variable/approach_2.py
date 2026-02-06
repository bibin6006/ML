import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

data=pd.read_csv('house.csv')

y=data.Price
features=data.drop(['Price'],axis=1)
missing_col=[col for col in features.columns if features[col].isnull().any()]
features=features.drop(missing_col,axis=1)

categorical_col=[col for col in features.columns if features[col].nunique()<10 and features[col].dtype=='object']
numarical_col=[col for col in features.columns if features[col].dtype in['int64','float64']]

x=features[categorical_col+numarical_col]

train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
new_train_x=train_x.copy()
new_val_x=val_x.copy()

OE=OrdinalEncoder()
new_train_x[categorical_col]=OE.fit_transform(train_x[categorical_col])
new_val_x[categorical_col]=OE.transform(val_x[categorical_col])

model=RandomForestRegressor(n_estimators=100,random_state=0)
model.fit(new_train_x,train_y)
predict_x=model.predict(new_val_x)
print(val_y.head())
print(predict_x[:5])
error=mean_absolute_error(val_y,predict_x)
print(error)
