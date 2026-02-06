import pandas as pd
from sklearn.tree import DecisionTreeRegressor
data=pd.read_csv('house.csv')
y=data.Price

features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=data[features]

model=DecisionTreeRegressor(random_state=1)
model.fit(x,y)
# print(y)
# print(model.predict(x.head()))

# predicting new house price 

new_house_features = {
    'Rooms': [5],
    'Bathroom': [3.0],
    'Landsize': [600.0],
    'Lattitude': [-37.78],
    'Longtitude': [145.02]
}
new_features=pd.DataFrame(new_house_features)
print(model.predict(new_features))
