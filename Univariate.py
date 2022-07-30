import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Reading data
data = pd.read_csv(r"D:/eng mariam/4th/ML/Tasks/Task2/univariateData.dat", sep=',',header=None)
print(data)

# Normalizing data
# scalar= preprocessing.MinMaxScaler()
# N = scalar.fit_transform(data)
# normalized_data = pd.DataFrame(N)
# print(normalized_data)

# Extraxting features and targets
features = data.values[:,0]
target = data.values[:,1]
num_samples=len(target)

# Splitting data
train_data,test_data,train_labels,test_labels= train_test_split(features,target,test_size=.2,random_state=42)

X_0_train = np.ones((len(train_data), 1))
X_1_train = train_data.reshape(len(train_data), 1)
X_0_test = np.ones((len(test_data), 1))
X_1_test = test_data.reshape(len(test_data), 1)
X_train = np.hstack((X_0_train, X_1_train))
X_test = np.hstack((X_0_test, X_1_test))
theta = np.zeros(2)

def fit(X,Y,theta):
    theta = Gradient_Descent(X,Y,theta)
    return theta 

def predict(X,theta):
    Y = X.dot(theta)
    return Y

def Compute_Cost(X,Y,theta):
    if len(X) != len(Y):
        raise TypeError("x and y should have same number of rows.")
    else:
        j = (1 / (2 * num_samples)) * np.sum(np.square(np.subtract(predict(X,theta), Y)))
    return j
def Gradient_Descent(X, Y, theta, alpha = 0.001, iterations = 1500):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        sum_delta = (alpha / num_samples) * X.transpose().dot(np.subtract(predict(X,theta), Y))
        theta = theta - sum_delta

        cost_history[i] = Compute_Cost(X, Y, theta)  
    print(cost_history)
    return theta

def Evaluate_Performance(Y_Predicted,Y_true):
    # err = Compute_Cost(X,Y,theta)
    # RMSE=np.sqrt(err)
    # acc = (1-err) * 100
    error=[]
    for i in range(len(Y_Predicted)) :
        diff=np.subtract(Y_Predicted[i], Y_true[i])
        error.append(abs(diff/Y_true[i]))
    accuracy=(np.sum(error)/len(Y_Predicted))*100
    return accuracy

# Testing Univariate Linear regression
theta = fit(X_train,train_labels,theta)
Predicted_labels = predict(X_test,theta)
Accuracy = Evaluate_Performance(Predicted_labels,test_labels)
print(Accuracy)