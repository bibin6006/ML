import pandas as pd
data=pd.read_csv('review.csv')

# for col in data.columns:
#     print(data[col])


# finding columns with missing value

missing_col=[col for col in data.columns if data[col].isnull().any()]
print(missing_col)

    