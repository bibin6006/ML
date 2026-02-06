import pandas as pd

#the question asks to identify the best bargain wine from a dataset. A "bargain" is defined as 
#the wine with the highest points-to-price ratio. The task is to create a variable named bargain_wine
# and assign it the title of this wine.
# idxmax() returns the index of the first occurrence of the maximum value in the Series

reviews=pd.read_csv('reviews.csv')
points_to_price=reviews.points/reviews.price
#print(points_to_price)
bargain_idx=points_to_price.idxmax()
#print(bargain_idx)
print(reviews.loc[bargain_idx,'title'])
