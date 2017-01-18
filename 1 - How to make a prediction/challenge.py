import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('challenge_dataset.txt', header = None)
df.columns = ['x','y']
x = df[[0]]
y = df[[1]]

reg = linear_model.LinearRegression()
reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,reg.predict(x))
plt.show()

x_test = df['x'][0]
y_test = df['y'][0]
y_predicted = reg.predict(x_test)[0][0]

print 'y = ', y_test
print 'y predicted = ', y_predicted
print 'error = ', y_predicted - y_test
