from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

df= pd.read_csv("Breast_cancer_data.csv")
x = df.loc[:, df.columns != 'diagnosis'].values #returns a numpy array
min_max_scaler = preprocessing.StandardScaler()
x_scaled = min_max_scaler.fit_transform(x)
diagnosis= df['diagnosis']
df = pd.DataFrame(x_scaled)
df['diagnosis']= diagnosis
model= KMeans(n_clusters= 2)
model.fit(df.loc[:, df.columns != 'diagnosis'])
labels = model.labels_
clusters= []
clusters.append([])
clusters.append([])

for index, row in df.iterrows():
  if labels[index] == 0:
    clusters[0].append(row)
  else:
    clusters[1].append(row)

for idx in range(len(clusters)):
  positive = 0
  negative = 0
  cluster = clusters[idx]
  for item in cluster:
    item= item.tolist()
    if item[-1]== 1.0:
      positive += 1
    else:
      negative += 1
  print("**" + str(positive) + ":" + str(negative))
