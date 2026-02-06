import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data=pd.read_csv('house.csv')

y=data.Price
features=data.drop(['Price'],axis=1)
x=features.select_dtypes(exclude=['object','bool'])
train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
train_x_plus=train_x.copy()
val_x_plus=val_x.copy()
missing_col=[col for col in train_x.columns if data[col].isnull().any()]

for col in missing_col:
    train_x_plus[col+'_was_missing']=train_x[col].isnull()
    val_x_plus[col+'_was_missing']=val_x[col].isnull()

my_impute=SimpleImputer()

imputer_train_x=pd.DataFrame(my_impute.fit_transform(train_x_plus))
imputer_val_x=pd.DataFrame(my_impute.transform(val_x_plus))
imputer_train_x.columns=train_x_plus.columns
imputer_val_x.columns=val_x_plus.columns
model=RandomForestRegressor(n_estimators=10,random_state=0)
model.fit(imputer_train_x,train_y)
predict_x=model.predict(imputer_val_x)
error=mean_absolute_error(val_y,predict_x)
print(error)