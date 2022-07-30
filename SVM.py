import pandas as pd
# import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing


data = pd.read_csv(r"D:/eng mariam/4th/ML/Tasks/Task4/data.txt",sep=' ',header=None)
# print(data)

features = data.values[:,:-1]
target = data.values[:,-1:]

scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(features)
scaled_df = pd.DataFrame(d) 

std = preprocessing.StandardScaler()
scaled_data = std.fit_transform(scaled_df) 


accuracies = []
for i in range(10):
    train_data,test_data,train_labels,test_labels= train_test_split(scaled_data,target,test_size=.4)

    svm_clf = svm.SVC(kernel="linear")
    svm_clf.fit(train_data, train_labels)
    pred_labels = svm_clf.predict(test_data)
    C_M = confusion_matrix(test_labels,pred_labels)
    TN=C_M[0][0]  
    FP=C_M[0][1]  
    FN=C_M[1][0]  
    TP=C_M[1][1]
    accuracy =  (TP+TN) /(TP+FP+TN+FN)
    accuracies.append(accuracy)

avg_acc = sum(accuracies) / len(accuracies)
print("The average accuracy after normalization and standardization is: ", avg_acc)

