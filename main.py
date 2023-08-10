import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LR import LinearRegression
dataSet = pandas.read_csv('car_data.csv')
# Seperatting x,y values "features and output"
prices = dataSet['price']
features = pandas.read_csv('car_data_selected_features.csv')
# print(features)

# Normalizing Data set
# features = features.apply(lambda x:(x - x.min(axis=0)) / (x.max(axis=0))-x.min(axis=0))
features = features.apply(lambda x:(x - x.min(axis=0)) / (x.max(axis=0)))
# print(features)

# Shuffle and splitting data
x_train, x_test, y_train, y_test = train_test_split(features,prices,test_size=0.2,random_state=5)


reg = LinearRegression(Alpha=0.01)
reg.fit(x_train,y_train)
predictions = reg.predict(x_test)
print(predictions)
# Cost function on testing
def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

print(mse(y_test,predictions))

# plt.scatter(dataSet.price, dataSet.enginesize)  # positive
# plt.show()
#
# plt.scatter(dataSet.price, dataSet.citympg)  # negative
# plt.show()
# plt.scatter(dataSet.price, dataSet.horsepower)  # Positive
# plt.show()
# plt.scatter(dataSet.price, dataSet.citympg)  # negative
# plt.show()
#
# plt.scatter(dataSet.price, dataSet.highwaympg)  # negative
# plt.show()
# plt.scatter(dataSet.price, dataSet.stroke)
# plt.show()
# plt.scatter(dataSet.price, dataSet.boreratio)  # maybe
# plt.show()
# plt.scatter(dataSet.price, dataSet.fuelsystem)
# plt.show()
#
# plt.scatter(dataSet.price, dataSet.cylindernumber)
# plt.show()
# plt.scatter(dataSet.price, dataSet.carheight)  # positive
# plt.show()
# plt.scatter(dataSet.price, dataSet.carwidth)  # positive
# plt.show()
# plt.scatter(dataSet.price, dataSet.fueltypes)
# plt.show()
# plt.scatter(dataSet.price, dataSet.aspiration)
# plt.show()
# plt.scatter(dataSet.price, dataSet.doornumbers)
# plt.show()
# plt.scatter(dataSet.price, dataSet.carbody)
# plt.show()
# plt.scatter(dataSet.price, dataSet.drivewheels)
# plt.show()
# plt.scatter(dataSet.price, dataSet.enginetype)
# plt.show()
