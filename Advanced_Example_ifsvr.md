# Gerekli Libleri Import Etme

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

# Dataseti Tanımlama Gereksiz Kolonları Temizleme Ve Ismini Değiştirme

```
df = pd.read_csv("One_year_compiled.csv")

df =df.drop(['timestamp', 'sample_Number', 'hour', 'day', 'month'], axis = 1)
df.rename(columns={"pCut::Motor_Torque" : "pCut_Motor_Torque"},inplace=True)
df.rename(columns={"pCut::CTRL_Position_controller::Lag_error" : "pCut_CTRL_Position_controller_Lag_error"},inplace=True)
```

# Isolation Forest Tanımlama Ve Onbinerli Gruplara Bölme

```
model = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),max_features=1.0)
model.fit(df[["pCut_Motor_Torque"]])
df['torq_anomalies_score'] = model.decision_function(df[["pCut_Motor_Torque"]])
df['torq_anomaly'] = model.predict(df[["pCut_Motor_Torque"]])

motor_torq_df = df[['torq_anomaly']].copy()

motor_torq_df['pCut_Motor_Torque'] = df['pCut_Motor_Torque']

motor_torq_df['torq_anomalies_score'] = df['torq_anomalies_score']

for i, group in df.groupby(np.arange(len(df)) // 10000):
    group.to_csv(f"group_{i}.csv", index=False)

df2 = pd.read_csv("group_0.csv")
```

# Train, Test, Split Oluşturma

```
from sklearn.model_selection import train_test_split

train, test = train_test_split(df2, test_size=0.2, random_state=42)

train = train.sort_values("pCut_Motor_Torque")
test = test.sort_values("pCut_Motor_Torque")

X_train, X_test = train[["pCut_Motor_Torque"]], test[["pCut_Motor_Torque"]]
y_train, y_test = train["torq_anomaly"], test["torq_anomaly"]
```

# Olçeklendirme Ve Fit Etme 
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVR

svr_lin = SVR(kernel = 'linear')
svr_rbf = SVR(kernel = 'rbf')
svr_poly = SVR(kernel = 'poly')

svr_lin.fit(X_train_scaled, y_train)
svr_rbf.fit(X_train_scaled, y_train)
svr_poly.fit(X_train_scaled, y_train)
```

# Predict Etme Ve Görselleştirme

```
from matplotlib import pyplot as plt

train['linear_svr_pred'] = svr_lin.predict(X_train_scaled)
train['rbf_svr_pred'] = svr_rbf.predict(X_train_scaled)
train['poly_svr_pred'] = svr_poly.predict(X_train_scaled)


plt.plot(train["pCut_Motor_Torque"], train['linear_svr_pred'], color = 'orange', label = 'Linear SVR')
plt.plot(train["pCut_Motor_Torque"], train['rbf_svr_pred'], color = 'green', label = 'Rbf SVR')
plt.plot(train["pCut_Motor_Torque"], train['poly_svr_pred'], color = 'blue', label = 'Poly SVR')
plt.legend()
plt.show()
```

<img width="562" alt="Ekran Resmi 2024-04-15 14 51 55" src="https://github.com/buzzi0/Task_Examples/assets/103946477/eb2ac56f-781a-4c68-8f10-f6855e88ae50">



