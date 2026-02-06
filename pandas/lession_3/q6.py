import pandas as pd 

reviews=pd.read_csv('reviews.csv')
#.str used to give string method to alll the value in the series 
#lowerr=reviews.description.str.lower()
#print(lowerr)

#description=lowerr.value_counts()

#print(description)

#res=description.loc[['tropical','fruity']]

#print(res)

#There are only so many words you can use when describing a bottle of wine. Is a wine more 
# likely to be "tropical" or "fruity"? Create a Series descriptor_counts counting how many
#  times each of these two words appears in the description column in the dataset. (For simplicity,
#  let's ignore the capitalized versions of these words.)

tropical_count=reviews.description.map(lambda desc: desc=='tropical').sum()
#print(tropical_count)
fruity_count=reviews.description.map(lambda desc: desc=='fruity').sum()
descriptor_counts=pd.Series([tropical_count,fruity_count],index=['tropical','fruity'])
print(descriptor_counts)
