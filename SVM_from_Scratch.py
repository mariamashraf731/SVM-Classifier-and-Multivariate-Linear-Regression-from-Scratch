import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
 
class LinearSVMUsingSoftMargin:
    def __init__(self, C=1.0):
        self._support_vectors = None
        self.C = C
        self.beta = None
        self.b = None
        self.X = None
        self.y = None
 
        # n is the number of data points
        self.n = 0
 
        # d is the number of dimensions
        self.d = 0
 
    def Decision_function(self, X):
        return X.dot(self.beta) + self.b
 
    def Cost(self, margin):
        return (1 / 2) * self.beta.dot(self.beta) + self.C * np.sum(np.maximum(0, 1 - margin))
 
    def Margin(self, X, y):
        return y * self.Decision_function(X)
 
    def fit(self, X, y, lr=1e-3, epochs=500):
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
        # Required only for plotting
        self.X = X
        self.y = y
        loss_array = []
        for _ in range(epochs):
            margin = self.Margin(X, y)
            loss = self.Cost(margin)
            loss_array.append(loss)
 
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * d_beta
 
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
            self._support_vectors = np.where(self.Margin(X, y) <= 1)[0]

    def plot_decision_boundary(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap=plt.cm.Paired, alpha=.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
 
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.Decision_function(xy).reshape(XX.shape)
 
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
 
        # highlight the support vectors
        ax.scatter(self.X[:, 0][self._support_vectors], self.X[:, 1][self._support_vectors], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
 
        plt.show()

    def predict(self, X):
        return np.sign(self.Decision_function(X))
 
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)
 
 
 
 
if __name__ == '__main__':


    data = load_iris()
    data.data = data.data[:,:2]
    
    for i in range(3):
        features = data.data[np.where(data.target!=i)]
        target = data.target[np.where(data.target!=i)]
        target[:50]= -1
        target[50:]= 1
        # scale the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        train_data,test_data,train_labels,test_labels= train_test_split(scaled_features,target,test_size=.2)
        model = LinearSVMUsingSoftMargin(C=15.0)
        model.fit(train_data, train_labels)
        print("train score:", model.score(train_data, train_labels))
        print("test score:", model.score(test_data, test_labels))
        model.plot_decision_boundary()


