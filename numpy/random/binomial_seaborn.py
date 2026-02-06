from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

#sns.displot(random.binomial(n=50,p=0.5,size=100))
sns.displot(random.binomial(n=50,p=0.5,size=100),kind="kde")
plt.show()

