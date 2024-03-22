# Task_Examples

🥇 Visualization
---
#Gerekli araçları yüklüyoruz
---
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import IsolationForest
```
#Dataseti tanıtıp gereksiz sütunları atıyoruz ve yeniden adlandırma yapıyoruz
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
#Isolation forest ile elde ettiğimiz birkaç sütunu kullanabilmek için Isolation Forest algoritmasını da import ediyoruz
---
```
model = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),max_features=1.0)
model.fit(df_clear[["pCut_Motor_Torque"]])
df_clear['torq_anomalies_score'] = model.decision_function(df_clear[["pCut_Motor_Torque"]])
df_clear['torq_anomaly'] = model.predict(df_clear[["pCut_Motor_Torque"]])

df_clear['pCut_Current_Position'] = df_clear['pCut_CTRL_Position_controller_Actual_position'] - df_clear['pCut_CTRL_Position_controller_Lag_error']
df_clear['pSvolFilm_Current_Position'] = df_clear['pSvolFilm_CTRL_Position_controller_Actual_speed'] - df_clear['pSvolFilm_CTRL_Position_controller_Lag_error']

motor_torq_df = df_clear[['torq_anomaly']].copy()

motor_torq_df['pCut_Motor_Torque'] = df_clear['pCut_Motor_Torque']

motor_torq_df['torq_anomalies_score'] = df_clear['torq_anomalies_score']
```

# 🥇Visualizations #
---
#Motor torq degerlerinin noktasal dagilimi
---
```
sns.scatterplot(df_clear.pCut_Motor_Torque)
plt.xlabel("Index")
plt.ylabel("Motor Torque")
plt.title("Motor torq degerlerinin noktasal dagilimi")
plt.show()
```
<img width="592" alt="1" src="https://github.com/buzzi0/Task_Examples/assets/103946477/dd801225-923f-4212-aba8-eac78333285a">

#Motor torq degerlerinin grafiksel dagilimi
---
```
sns.kdeplot(df_clear.pCut_Motor_Torque, shade = True)
plt.xlabel("Motor Torque")
plt.ylabel("Value Distribution")
plt.title("Motor torq degerlerinin grafiksel dagilimi")
plt.show()
```
<img width="579" alt="2" src="https://github.com/buzzi0/Task_Examples/assets/103946477/a19a4552-a7ba-4792-a55a-3dcee7afe86e">

#Normal ve anormal degerlerin dagilimi
---
```
sns.countplot(x='torq_anomaly', data=df_clear, palette='rocket')
plt.title('Normal ve anormal degerlerin dagilimi')
plt.show()
```

<img width="605" alt="3" src="https://github.com/buzzi0/Task_Examples/assets/103946477/6f80510f-b916-4be5-8d5e-bf55a7cf7323">

#Kesme Bıçağının Hıza Göre Pozisyon Değişimi
---
```
sns.scatterplot(data=df_clear, x='pCut_CTRL_Position_controller_Actual_speed', y='pCut_CTRL_Position_controller_Actual_position')
plt.xlabel('Hız')
plt.ylabel('Pozisyon')
plt.title('Hıza Göre Pozisyon Değişimi')
plt.grid(True)
plt.show()
```

<img width="590" alt="4" src="https://github.com/buzzi0/Task_Examples/assets/103946477/01f90374-f6e7-409d-bb4b-0e1d18ec86af">

#Plastik Film Açıcının Hıza Göre Pozisyon Değişimi
---
```
sns.scatterplot(data=df_clear, x='pSvolFilm_CTRL_Position_controller_Actual_speed', y='pSvolFilm_CTRL_Position_controller_Actual_position')
plt.xlabel('Hız')
plt.ylabel('Pozisyon')
plt.title('Plastik Film Çözücünün Hıza Göre Pozisyon Değişimi')
plt.grid(True)
plt.show()
```

<img width="585" alt="5" src="https://github.com/buzzi0/Task_Examples/assets/103946477/3f258cad-1256-46de-8b23-dbf5e6231a14">

#Total Mode Degerleri
---
```
sns.countplot(x='mode', data=df_clear, palette='rocket')
plt.title('Total Mod Degerleri')
plt.show()
```

<img width="598" alt="6" src="https://github.com/buzzi0/Task_Examples/assets/103946477/319dd615-4362-4931-8feb-310cbaa4c1e1">

#Total Mod Degerleri Pie Şeklinde
---
```
distribution = df_clear['mode'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(distribution, labels=distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Modeların Dağılımı')
plt.axis('equal')  
plt.show()
```
<img width="621" alt="7" src="https://github.com/buzzi0/Task_Examples/assets/103946477/67eb0de8-846c-4f5b-85aa-20225c649b21">

#Modelara Göre Performans Değerlerinin Dağılımı
---
```
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_clear, x='mode', y='pSpintor_VAX_speed', marker='o', color='blue', linewidth=2.5)
plt.title('Modelara Göre Performans Değerlerinin Dağılımı')
plt.xlabel('Model')
plt.ylabel('Performans')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
```

<img width="722" alt="8" src="https://github.com/buzzi0/Task_Examples/assets/103946477/eccee9ac-5f53-43d4-a34e-c5737b584f77">

#Modelara Göre Motor Torquesi Değişimi 
---
```
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_clear, x='mode', y='pCut_Motor_Torque', marker='o', color='blue', linewidth=2.5)
plt.title('Modelara Göre Motor Torquesi Değişimi')
plt.xlabel('Model')
plt.ylabel('Torq')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
```

<img width="720" alt="9" src="https://github.com/buzzi0/Task_Examples/assets/103946477/b0608216-83e9-42bd-8563-da62fcecf855">

#Hatalı Konum İle Doğru Konum Arasında ki Fark
---
```
sns.kdeplot(df_clear.pCut_Current_Position, color='blue', label='Data 1')
sns.kdeplot(df_clear.pCut_CTRL_Position_controller_Actual_position, color='red', label='Data 2')
plt.xlabel("Motor Torque")
plt.ylabel("Value Distribution")
plt.title("Motor torq degerlerinin grafiksel dagilimi")
plt.legend()
plt.show()
```

<img width="586" alt="10" src="https://github.com/buzzi0/Task_Examples/assets/103946477/eab5f155-11cf-4999-aeb6-bea5ccfa7738">

<img width="1413" alt="10-1" src="https://github.com/buzzi0/Task_Examples/assets/103946477/d0a55ceb-57e3-4eb5-9cbe-029b6ec9d9c4">

#Motor Tork Anomali Görselleştirme
---
```
plt.scatter(motor_torq_df['pCut_Motor_Torque'], motor_torq_df['torq_anomalies_score'], c=motor_torq_df['torq_anomaly'], cmap='coolwarm')
plt.xlabel('Motor Torq')
plt.ylabel('Anomaly Scores')
plt.title('Motor Tork Anomali Görselleştirme')
plt.colorbar(label='Anomali')
plt.show()
```

<img width="607" alt="11" src="https://github.com/buzzi0/Task_Examples/assets/103946477/98d71a4c-ec87-467f-bde7-727471db94a2">

#Motor Tork Değişimi ve Performans İlişkisi
---
```
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clear, x='pSpintor_VAX_speed', y='pCut_Motor_Torque')
plt.xticks('pSpintor_VAX_speed')
plt.ylabel('Performans')
plt.title('Motor Tork Değişimi ve Performans İlişkisi')
plt.show()
```

<img width="887" alt="12" src="https://github.com/buzzi0/Task_Examples/assets/103946477/ba14b98d-78fb-4d32-8e17-b56c204587a6">


