import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('house.csv')
y=data.Price
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=data[features]

train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8,test_size=0.2)

model_1=RandomForestRegressor(n_estimators=50,random_state=0)
model_2=RandomForestRegressor(n_estimators=100,random_state=0)
model_3=RandomForestRegressor(n_estimators=100,criterion='absolute_error',random_state=0)
model_4=RandomForestRegressor(n_estimators=200,min_samples_split=30,random_state=0)
model_5=RandomForestRegressor(n_estimators=100,max_depth=7,random_state=0)
models=[model_1,model_2,model_3,model_4,model_5]

def get_mae(model,train_x,val_x,train_y,val_y):
    model.fit(train_x,train_y)
    predict_x=model.predict(val_x)
    error=mean_absolute_error(val_y,predict_x)
    return error

for i in models:
    mae=get_mae(i,train_x,val_x,train_y,val_y)
    print(i,'mean_absolute_error:',mae)

