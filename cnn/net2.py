import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

np.random.seed(0)
X,y = sklearn.datasets.make_moons(200,noise=0.20)
def random_data_set():
    np.random.seed(0)
    X,y = sklearn.datasets.make_moons(200,noise=0.20)
    # plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    return X,y
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
def LogisticRegression():
    # X,y = random_data_set()
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X,y)
    
    plot_decision_boundary(lambda x:clf.predict(x))
    plt.title('Logistic Regression')
    plt.show()
if __name__=='__main__':
    LogisticRegression()


