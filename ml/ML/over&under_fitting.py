import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data=pd.read_csv('house.csv')

features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x=data[features]
y=data.Price
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)

def get_mae(max_leaf_node,train_x,val_x,tain_y,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_node,random_state=0) #max_leaf_nodes is count of leaf node , this adjust rhe tree's leaf node to max_leaf_nodes value 
    model.fit(train_x,train_y)
    prediction_x=model.predict(val_x)
    error=mean_absolute_error(val_y,prediction_x)
    return error

for max_leaf_node in [450,475,500]:
    mae=get_mae(max_leaf_node,train_x,val_x,train_y,val_y)

    print("max_leaf_node at:",max_leaf_node,"mean absolute error:",mae)
