import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv("house.csv")
y=data.Price 
features=data.drop(['Price'],axis=1)

missing_col=[col for col in features.columns if features[col].isnull().any()] 
features=features.drop(missing_col,axis=1)
categorical_col=[col for col in features.columns if features[col].nunique()<10 and features[col].dtypes=='object']

numerical_col=[col for col in features.columns if features[col].dtypes in ['int64','float64']]

x=features[categorical_col+numerical_col]

train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8,test_size=0.2)

oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False) #handle_output -> handle un-seened(new) data during transform() stage here handle_ouput ='ignore' so it remove(drop) un-seened(new) data 
#sparse_output=False return the data in numpy array 
oh_train_x_col=pd.DataFrame(oh.fit_transform(train_x[categorical_col]))
oh_val_x_col=pd.DataFrame(oh.transform(val_x[categorical_col]))
# print(len(oh_train_x_col.columns))
# print(len(oh_val_x_col.columns))
# print(len(categorical_col))
# print(oh_train_x_col.columns)
numerical_train_x=train_x.drop(categorical_col,axis=1)
numerical_val_x=val_x.drop(categorical_col,axis=1)
# print(numerical_train_x)
# print(numerical_val_x)
oh_train_x_col.index=numerical_train_x.index
oh_val_x_col.index=numerical_val_x.index

new_train_x=pd.concat([numerical_train_x,oh_train_x_col],axis=1)
new_Val_x=pd.concat([numerical_val_x,oh_val_x_col],axis=1)

new_train_x.columns=new_train_x.columns.astype(str)
new_Val_x.columns=new_Val_x.columns.astype(str)
# print(new_Val_x)
# print(new_train_x)


model=RandomForestRegressor(n_estimators=100,random_state=0)
model.fit(new_train_x,train_y)
predict_X=model.predict(new_Val_x)
error=mean_absolute_error(val_y,predict_X)
print(val_y.head())
print(predict_X[:5])
print(error)




