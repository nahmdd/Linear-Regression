import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, Alpha = 0.001, n_iters=1000):
        self.Alpha = Alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,x,y):
        n_samples, n_features = x.shape
        self.weights=np.zeros(n_features)
        self.bias = 0
        cost = [None] * self.n_iters
        iterations = [None] * self.n_iters
        for i in range(self.n_iters):
            y_predict = np.dot(x, self.weights) + self.bias
            iterations[i] = i
            cost[i] = np.mean((y-y_predict)**2)
            dw = (1 / n_samples) * np.dot(x.T, (y_predict - y))
            db = (1 / n_samples) * np.sum(y_predict - y)
            self.weights = self.weights - self.Alpha * dw
            self.bias = self.bias - self.Alpha * db
        plt.scatter(iterations, cost, color="b", marker="o", s=30)
        plt.xlabel("Epoch (number of iteration)")
        plt.ylabel( "Cost or Loss")
        plt.show()


    def predict(self,x):
        return np.dot(x,self.weights) + self.bias
