import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


col_names = ["pregnant","glucose","bp","skin","insulin","bmi","pedigree","age","label"]
cwd = os.getcwd()
data_dir = "/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP 499/Datasets"
os.path.join(data_dir, "diabetes.csv")
diabetes_knn = pd.read_csv(os.path.join(data_dir, "diabetes.csv"), header=1, names=col_names)

#print(diabetes_knn)

# print(diabetes_knn.isnull().any())
#
# print(diabetes_knn.info())
#
feature_cols = ["pregnant","glucose","bp","skin","insulin","bmi","pedigree","age"]
X=diabetes_knn[feature_cols]
y=diabetes_knn["label"]
#print(X.head())
#print(y.head())

scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
# print(X)

X_trainA, X_temp, y_trainA, y_temp = train_test_split(X,y, test_size=0.4, random_state=2021, stratify=y)
X_trainB, X_test, y_trainB, y_test = train_test_split(X_temp,y_temp, test_size=0.5, random_state=2021, stratify=y_temp)

neighbors = np.arange(1,31)
trainA_accuracy = np.empty(30)
trainB_accuracy = np.empty(30)

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA,y_trainA)
    y_pred = knn.predict(X_trainB)
    cf = metrics.confusion_matrix(y_trainB, y_pred)
    trainA_accuracy[k-1] = knn.score(X_trainA, y_trainA)
    trainB_accuracy[k-1] = knn.score(X_trainB, y_trainB)

# plt.figure(2)
# plt.title("KNN: Varying Number of Neighbors")
# plt.plot(neighbors, trainB_accuracy, label="Training B Accuracy")
# plt.plot(neighbors, trainA_accuracy, label="Training A Accuracy")
# plt.legend()
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.show()

test_accuracy = np.empty(30)

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA,y_trainA)
    y_pred = knn.predict(X_trainB)
    cf = metrics.confusion_matrix(y_trainB, y_pred)
    test_accuracy[k-1] = knn.score(X_test, y_test)

# plt.figure(1)
# plt.title("KNN: Varying Number of Neighbors")
# plt.plot(neighbors, test_accuracy, label="Test Accuracy")
# plt.legend()
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.show()

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_test,y_test)
#print(knn.score(X_test, y_test))

metrics.plot_confusion_matrix(knn,X_test,y_test)