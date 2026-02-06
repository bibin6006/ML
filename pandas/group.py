import pandas as pd
reviews=pd.read_csv('reviews.csv')
#print(reviews)

#The groupby() method groups rows that have the same value based on a specified parameter. In 
# this case, 'points' is the parameter, and the method checks for rows with identical 'points' values to group them.

group=reviews.groupby('points').count()

print(group)

#group=reviews.groupby('points')
#print(group.price.min()) # it print the minimum value of each group 

# to find the first title of every group

#print(reviews.groupby('winery').apply(lambda x:x.title.iloc[0]))



