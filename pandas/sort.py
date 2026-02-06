import pandas as pd
reviews=pd.read_csv('reviews.csv')

# sorted=reviews.sort_values(by='price')
# print(sorted)

# print(reviews.sort_values(by='price',ascending=False))

# print(reviews.sort_values(by=['points','price']))

# print(reviews.sort_values(by=['points','price'],ascending=[True,False]))

print(reviews.sort_index())