import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# load data
df = pd.read_csv('qatarcars.csv')
df = df.drop(columns = ['make', 'model', 'width', 'height', 'seating', 'trunk', 'length'])

# encode data for corr matrix
df_encoded = pd.get_dummies(df, drop_first=True)
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.show()

# distribution for outlier removal
print(df['price'].describe())

# remove outliers
price_filtered = df[df['price'] < 5500000]

# boxplot of distributions
plt.boxplot(np.log(price_filtered['price']))
plt.show()

# histogram of distributions
plt.hist(np.log(price_filtered['price']), bins=20, edgecolor='black')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.title('Distribution of Log(Price)')
plt.show()
