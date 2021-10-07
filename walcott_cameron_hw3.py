#Cameron Walcott
#ITP 499 Fall 2021
#HW3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import seaborn as sb
from sklearn.linear_model import LogisticRegression

cwd = os.getcwd()
col_names = ["Passenger", "Ticket Class", "Sex", "Age", "Survival"]
data_dir = "/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP 499/Datasets"
os.path.join(data_dir, "titanic.csv")
titanic = pd.read_csv(os.path.join(data_dir, "titanic.csv"), header=1, names=col_names)


#print(titanic)


#titanic.info()
#print(titanic.isnull().any())


titanic.drop('Passenger', axis=1, inplace=True)
#print(titanic)


plt.figure(1)
sb.countplot(x="Survival", data=titanic)

plt.figure(2)
sb.countplot(y="Ticket Class", data=titanic)

plt.figure(3)
sb.countplot(y="Age", data=titanic)

plt.figure(4)
sb.countplot(x="Sex", data=titanic)

#plt.show()

titanic2 = pd.get_dummies(titanic, columns=["Ticket Class", "Sex", "Age", "Survival"])
#print(titanic2)


X = titanic2.iloc[:,:8]
y = titanic2.iloc[:,9]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2021, stratify=y)

# X_train.shape
#y_train.shape

#X_test.shape
#y_test.shape


logReg = LogisticRegression()
logReg.fit(X_train,y_train)


y_pred = logReg.predict(X_test)
# plt.figure(5)
# skplt.metrics.plot_lift_curve(y_test,y_pred)
# plt.show()

# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# #logReg.classes_
# #cnf_matrix
# print(metrics.accuracy_score(y_test, y_pred))
#
# disp = metrics.ConfusionMatrixDisplay(cnf_matrix, display_labels=['Yes','No'])
# disp.plot()
# plt.show()


titanic_pred = titanic
titanic_pred.drop('Survival', axis=1, inplace=True)
new = {'Ticket Class': ['3rd'], 'Sex': ['Male'], 'Age': ['Adult']}
new = pd.DataFrame(new)
titanic_pred = titanic_pred.append(new, ignore_index=True)
titanic_pred = pd.get_dummies(titanic_pred, columns=["Ticket Class", "Sex", "Age"])

predictions = logReg.predict(titanic_pred)
# indexing the last element of the predictions array since
# the passenger who we are predicting for was added to
# the end of the dataframe of passengers
print(predictions[2200])
