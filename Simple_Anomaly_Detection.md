#Anomaly Detection#
---

#Gerekli araçları yükleme
---
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
```

#Veri setini tanımlama gereksiz sütunları atma ve yeniden adlandırma
---
```
df = pd.read_csv("One_year_compiled.csv")

df_clear =df.drop(['timestamp', 'sample_Number', 'hour', 'day', 'month'], axis = 1)
df_clear.rename(columns={"pCut::Motor_Torque" : "pCut_Motor_Torque"},inplace=True)
df_clear.rename(columns={"pCut::CTRL_Position_controller::Lag_error" : "pCut_CTRL_Position_controller_Lag_error"},inplace=True)
df_clear.rename(columns={"pCut::CTRL_Position_controller::Actual_position" : "pCut_CTRL_Position_controller_Actual_position"},inplace=True)
df_clear.rename(columns={"pCut::CTRL_Position_controller::Actual_speed" : "pCut_CTRL_Position_controller_Actual_speed"},inplace=True)
df_clear.rename(columns={"pSvolFilm::CTRL_Position_controller::Actual_position" : "pSvolFilm_CTRL_Position_controller_Actual_position"},inplace=True)
df_clear.rename(columns={"pSvolFilm::CTRL_Position_controller::Actual_speed" : "pSvolFilm_CTRL_Position_controller_Actual_speed"},inplace=True)
df_clear.rename(columns={"pSvolFilm::CTRL_Position_controller::Lag_error" : "pSvolFilm_CTRL_Position_controller_Lag_error"},inplace=True)
df_clear.rename(columns={"pSpintor::VAX_speed" : "pSpintor_VAX_speed"},inplace=True)
```

🥇Isolation Forest
---

```
model = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),max_features=1.0)
model.fit(df_clear[["pCut_Motor_Torque"]])
df_clear['torq_anomalies_score'] = model.decision_function(df_clear[["pCut_Motor_Torque"]])
df_clear['torq_anomaly'] = model.predict(df_clear[["pCut_Motor_Torque"]])

motor_torq_df = df_clear[['torq_anomaly']].copy()

motor_torq_df['pCut_Motor_Torque'] = df_clear['pCut_Motor_Torque']

motor_torq_df['torq_anomalies_score'] = df_clear['torq_anomalies_score']

plt.scatter(motor_torq_df['pCut_Motor_Torque'], motor_torq_df['torq_anomalies_score'], c=motor_torq_df['torq_anomaly'], cmap='coolwarm')
plt.xlabel('Motor Torq')
plt.ylabel('Anomaly Scores')
plt.title('Motor Tork Anomali Görselleştirme')
plt.colorbar(label='Anomali')
plt.show()
```


<img width="607" alt="11" src="https://github.com/buzzi0/Task_Examples/assets/103946477/b0aea7f4-0b8a-4f7f-a770-0821b6c451fe">




<img width="409" alt="Ekran Resmi 2024-03-22 14 57 06" src="https://github.com/buzzi0/Task_Examples/assets/103946477/5d1b895d-6860-4d92-afed-36c0fdcafe93">



🥈K-Means
---

```
X = df_clear

scaler = preprocessing.MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), 
                              columns=X.columns, 
                              index=X.index)

X = preprocessing.scale(X)


train_percentage = 0.15
train_size = int(len(df_clear.index)*train_percentage)
X_train = X[:train_size]


kmeans = KMeans(n_clusters=1)

kmeans.fit(X_train)

k_anomaly = df_clear.copy()

k_anomaly = pd.DataFrame(kmeans.transform(X))

k_anomaly.to_csv('KM_Distance.csv')

plt.subplots(figsize=(15,7))

plt.plot(k_anomaly.index, k_anomaly[0], 'g', markersize=1)
```

<img width="1289" alt="Ekran Resmi 2024-03-22 14 53 32" src="https://github.com/buzzi0/Task_Examples/assets/103946477/300e47fa-8f7e-4c7c-a13b-3de9d6045822">


🥉KNN 
---

```

k = 3  
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(df_clear)

distances, indices = knn_model.kneighbors(df_clear)

avg_distances = np.mean(distances, axis=1)

threshold = np.percentile(avg_distances, 95)  # %95'lik eşik değeri
knn_anomalies = df_clear[avg_distances > threshold]

print("knn_Anomali Noktaları:")
print(knn_anomalies)
```

<img width="529" alt="Ekran Resmi 2024-03-22 14 55 45" src="https://github.com/buzzi0/Task_Examples/assets/103946477/bb5a1fb2-edc0-4cbe-bfe8-bd6ed36a09ce">

