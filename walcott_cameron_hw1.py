# Cameron Walcott
# ITP 499 Fall 2021
# HW1

import pandas as pd
import matplotlib.pyplot as plt
import os.path
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


cwd = os.getcwd()
data_dir = "/Users/cameronwalcott/Dropbox/Mac/Desktop/ITP 499"
os.path.join(data_dir, "wineQualityReds.csv")
wine = pd.read_csv(os.path.join(data_dir, "wineQualityReds.csv"))

wine.drop('Wine', axis=1, inplace=True)


quality = wine['quality']
wine.drop('quality', axis=1, inplace=True)

print(wine)
print(quality)

norm = Normalizer()
wine_normK = pd.DataFrame(norm.transform(wine), columns=wine.columns)
print(wine_normK)

ks = range(1, 11)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(wine_normK)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
plt.show()



model = KMeans(n_clusters=6, random_state=2021)
model.fit(wine_normK)
labels = model.predict(wine_normK)
wine_normK["cluster Label"] = pd.Series(labels)
print(wine_normK)
print(plt.hist(labels))


wine_normK["quality"] = quality

print(pd.crosstab(wine_normK["quality"], wine_normK["cluster Label"], values=None, rownames=None, colnames=None))
