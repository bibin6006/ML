import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data=pd.read_csv('house.csv')

features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=data[features]
y=data['Price']
model=DecisionTreeRegressor()
model.fit(x,y)
predict_x=model.predict(x)
error=mean_absolute_error(y,predict_x) #find the error( y-predict_x )is the error if y and predict_x is series then the average is error
print(error)