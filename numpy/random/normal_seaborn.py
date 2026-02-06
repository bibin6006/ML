from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

#sns.displot(random.normal(size=5))

#sns.displot(random.normal(size=10000),kind="kde")

sns.displot(random.normal(loc=10,scale=5,size=10000),kind="kde")

plt.show()