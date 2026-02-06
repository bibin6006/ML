import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('house.csv')
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=data[features]
y=data.Price

train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)

model=DecisionTreeRegressor()
model.fit(train_x,train_y)
predict_val_x=model.predict(val_x)

error=mean_absolute_error(val_y,predict_val_x)
print(error)