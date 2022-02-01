# Cameron Walcott
# ITP 499 Fall 2021
# Final Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contractions
import nltk
import regex as re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


tsv_file = '/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP499/Final Project/amazon_reviews.tsv'
review_data = pd.read_csv("/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP499/Final Project/amazon_reviews.tsv",
                          sep='\t', on_bad_lines='skip')
# print(review_data.head(10))


selected = ['review_body', 'star_rating']
reviews = review_data[selected]
# print(reviews.head(10))
print(len(reviews))

# Change positive values to 1
p = reviews[reviews.loc[:,'star_rating'] > 3]
positive = p.sample(n = 100000, replace = False)
positive = positive.reset_index(drop=True)
positive['star_rating'] = 1
print(positive)

# Change negative values to 0
n = reviews[reviews.loc[:,'star_rating'] < 3]
negative = n.sample(n = 100000, replace = False)
negative = negative.reset_index(drop=True)
negative['star_rating'] = 0
print(negative)


total_reviews = positive.append(negative)
total_reviews = total_reviews.sample(frac=1).reset_index(drop=True)
print(total_reviews.head(3))


total_words = 0
for i in range(len(total_reviews)):
    sentence = str(total_reviews['review_body'][i])
    words = sentence.split(" ")
    total_words = total_words + len(words)

print(total_words / 200000)

# lowercase
total_reviews['review_body'] = total_reviews['review_body'].str.lower()
print("lowercase")
print(total_reviews.head(3))

# urls and html
total_reviews['review_body'] = total_reviews['review_body'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value=' ',regex=True)
total_reviews['review_body'] = total_reviews['review_body'].str.replace(r'<[^<>]*>', ' ', regex=True)
print("urls and html")
print(total_reviews.head(3))

# contractions
total_reviews['review_body'] = \
    total_reviews['review_body'].apply(lambda x: [contractions.fix(str(word)) for word in str(x).split()])
print("Contractions")
print(total_reviews.head(3))

total_reviews['review_body'] = total_reviews['review_body'].apply(lambda x: ' '.join(x))


total_reviews['review_body'] = total_reviews['review_body'].str.replace('[^\w\s]',' ')
total_reviews['review_body'] = total_reviews['review_body'].str.replace('\d+',' ')
total_reviews['review_body'] = total_reviews['review_body'].str.replace('_','')
print("non alphabetic characters")
print(total_reviews.head(3))

stop_words = set(stopwords.words('english'))
total_reviews['review_body'] = \
    total_reviews['review_body'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop_words)]))
print("stop words")
print(total_reviews.head(3))


lemmatizer = WordNetLemmatizer()

total_reviews['review_body'] = \
    total_reviews['review_body'].apply(lambda x: [lemmatizer.lemmatize(str(word)) for word in str(x).split()])
total_reviews['review_body'] = total_reviews['review_body'].apply(lambda x: ' '.join(x))
print("lemmatizer")
print(total_reviews.head(3))

total_words2 =  0
for i in range(len(total_reviews)):
    sentence = str(total_reviews['review_body'][i])
    words = sentence.split(" ")
    total_words2 = total_words2 + len(words)

print(total_words2 / 200000)
print("2nd count")
print(total_reviews.head(3))

X = total_reviews['review_body']
y = total_reviews['star_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(vectorizer.get_feature_names())



clf = MultinomialNB()
clf.fit(X_train, y_train)
print("Accuracy on training data: ", clf.score(X_train, y_train))

y_pred = clf.predict(X_test)
print("Accuracy on testing data: ", metrics.accuracy_score(y_test, y_pred))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
matrix = metrics.ConfusionMatrixDisplay(cnf_matrix, display_labels=["Negative", "Positive"])
matrix.plot()
plt.title("Confusion Matrix")
plt.show()


print("Review Body: ", total_reviews['review_body'][2190])
print("Actual Sentiment: ", total_reviews['star_rating'][2190])
print("Predicted Sentiment: ", y_pred[2190])



user = input("please enter a review: ")
user = user.lower()
user = re.sub(r'[^\w\s]', '', user)
user = user.replace(r'<[^<>]*>', '')
user = contractions.fix(str(user))
user = user.replace('\d+','')
user = user.replace('_','')
user = user.split()
user = [word for word in user if not word in stopwords.words()]
user = ' '.join(user)
user = lemmatizer.lemmatize(user)
l = []
l.append(user)
user = pd.DataFrame(l)
vect = TfidfVectorizer()
user = vect.fit_transform(user[0])

prediction = clf.predict(user)
print(prediction)
print(X_test)