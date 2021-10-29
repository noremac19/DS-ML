import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

data_dir = "/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP 499/Datasets"
alphabet = pd.read_csv(os.path.join(data_dir,"alphabet_letters.csv"))
# print(alphabet.head())

words = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
         14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}


target = alphabet.iloc[:,:1]
features = alphabet.iloc[:,1:]
# print(target)
# print(features)


# print(features.shape)
# print(target.shape)
#
#
# plt.figure(1)
# sb.countplot(x="label", data=target)
# plt.show()
#
#
letter = features.iloc[1340,:]
letter = np.array(letter)
letter = letter.reshape(28,28)
l = words[target.iloc[1340,0]]

# plt.imshow(letter, cmap="gray")
# plt.title("The letter is " + str(l))
# plt.show()


X_train, X_test, y_train, y_test = \
    train_test_split(features,target,test_size=0.3, random_state=2021, stratify=target)


X_train = X_train / 255
X_test = X_test / 255


mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation="logistic", max_iter=5,
                    alpha=0.0001, solver="adam", random_state=2021, learning_rate_init=0.01, verbose=True)

# Train the network
mlp.fit(X_train, y_train)

#Display accuracy of the test data
print("The accuracy of the test data is: ", mlp.score(X_test, y_test))

# Display confusion matrix
plot_confusion_matrix(mlp, X_test, y_test)
plt.show()


y_pred = mlp.predict(X_test)
prediction = y_pred[0]

row1 = X_test.iloc[0,:]
row1 = np.array(row1)
row1 = row1.reshape(28,28)

plt.imshow(row1, cmap="gray")
plt.title("The predicted letter is " + str(words[prediction]) + " and the actual letter is " + str(words[y_test.iloc[0,0]]))
plt.show()



failed = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
# f_index = failed.sample(n=1).index
#
# sample = np.array(X_test.loc[f_index]).reshape(28,28)
#
# plt.imshow(sample, cmap="gray")
# plt.title("The failed predicted digit is: " + str(y_pred[f_index]) + ". The Actual digit is: " + str(int(y_test[f_index])))
# plt.show()


















