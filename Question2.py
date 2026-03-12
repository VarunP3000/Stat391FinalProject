import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("qatarcars.csv")

keep_features = [
    'origin', 'length', 'width', 'height', 'seating', 
    'trunk', 'economy', 'horsepower', 'mass', 
    'performance', 'type', 'enginetype'
]

formula = 'price ~ ' + ' + '.join(keep_features)

model = smf.ols(formula, data=df).fit()
print(model.summary())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.residplot(x=model.fittedvalues, y=model.resid)
plt.xlabel('Fitted Sales')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Sales')

plt.tight_layout()
plt.show()