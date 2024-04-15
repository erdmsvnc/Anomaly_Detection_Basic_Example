# Gerekli Libleri Import Etme

```
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from numpy.random import seed
from scipy.special import softmax
```

# Data Setleri Import Etme Ve Gerekli Sütun Düzenlemelerini Tamamlama

```
df_main = pd.read_csv("One_year_compiled.csv")


def handle_non_numeric(df):
    # Values in each column for each column
    columns = df.columns.values
    
    for column in columns:
        
        # Dictionary with each numerical value for each text
        text_digit_vals = {}
        
        # Receives text to convert to a number
        def convert_to_int (val):
            
            # Returns respective numerical value for class
            return text_digit_vals[val]
        
        # If values in columns are not float or int
        if df[column].dtype !=np.int64 and df[column].dtype != np.float64:
            
            # Gets values form current column
            column_contents = df[column].values.tolist()
            
            # Gets unique values from current column
            unique_elements = set(column_contents)
            
            # Classification starts at 0
            x=0
            
            for unique in unique_elements:
                
                # Adds the class value for the text in dictionary, if it's not there
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            
            # Maps the numerical values to the text values in columns 
            df[column] = list(map(convert_to_int, df[column]))
    
    return df

df_clear = df_main.drop(['day', 'sample_Number', 'month', 'timestamp'], axis=1)

df_clear = handle_non_numeric(df_clear)

X = df_clear
```

# Verileri normalize edip , ön işlemesini yapıp ölçeklendirme 

```
scaler = preprocessing.MinMaxScaler()
#Preprocessing
X = pd.DataFrame(scaler.fit_transform(X), 
                              columns=X.columns, 
                              index=X.index)


#Scaling
X = preprocessing.scale(X)
#Splitting the feature data for training data. First 200.000 rows.
X_train = X[:10000]
```

# OC SVM Modelini oluşturma 

```
X_train = X[:10000]
ocsvm = OneClassSVM(nu=0.25, gamma=0.05)
ocsvm.fit(X_train)

df=df_clear.copy()
df['anomaly'] = pd.Series(ocsvm.predict(X))

df.to_csv('Labled_df.csv')

df = pd.read_csv('Labled_df.csv', index_col=0)
df['Index'] = range(len(df))

#Getting labled groups
scat_1 = df.groupby('anomaly').get_group(1)
scat_0 = df.groupby('anomaly').get_group(-1)
```

# Aykırı değer oranını ve Doğruluk değerini hesaplama

```
# Aykırı değerlerin yüzdesini hesapla
anomali_orani = (df['anomaly'] == -1).sum() / len(df)

print("Aykırı Değer Oranı:", anomali_orani)

# Doğru tespit edilen aykırı değerlerin sayısını bulun
dogru_tespit_edilen = (scat_0['anomaly'] == -1).sum()

# Gerçek aykırı değerlerin sayısını bulun (etiketli test veri setinden)
gercek_aykirilar = len(scat_1[scat_1['anomaly'] == 1])

# Doğruluk değerini hesaplayın
dogruluk = 100- (dogru_tespit_edilen / gercek_aykirilar)

print("Doğruluk Değeri:",dogruluk)
```

# Görselleştirme

```
# Plot size
plt.subplots(figsize=(15,7))

# Plot group 1 -labeled, color green, point size 1
plt.plot(scat_1.index,scat_1['pCut::Motor_Torque'], 'r.', markersize=1)

# Plot group -1 -labeled, color red, point size 1
plt.plot(scat_0.index, scat_0['pCut::Motor_Torque'],'g.', markersize=1)

plt.show()
```

<img width="1215" alt="Ekran Resmi 2024-04-15 14 37 51" src="https://github.com/buzzi0/Task_Examples/assets/103946477/2c096e03-46c8-41c2-8af9-2e6837806baa">


```
df.rename(columns={"pCut::Motor_Torque" : "pCut_Motor_Torque"},inplace=True)



plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='pCut_Motor_Torque', data=df)
plt.title('Zamana Göre Motor Torkunun Değişimi')
plt.xlabel('hour')
plt.ylabel('Motor Torku')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

<img width="993" alt="Ekran Resmi 2024-04-15 14 38 43" src="https://github.com/buzzi0/Task_Examples/assets/103946477/9d46dab2-0151-4522-92e3-0114ab0d1378">

```
plt.hist(df.hour, bins = 20)
plt.title("Belirtilen saatte ki toplam veri sayısı")
plt.show()
```

<img width="584" alt="Ekran Resmi 2024-04-15 14 39 31" src="https://github.com/buzzi0/Task_Examples/assets/103946477/845137c2-2aa4-4f5d-9b25-2abf8520ea83">


```
sns.lineplot(
    x="Index",
    y="pCut_Motor_Torque",
    hue="anomaly",
    data=df,
    
)
plt.title("Motor Torqu'nun Index'e Göre Anomali Dağılımı")
plt.xlabel("Index")
plt.ylabel("Motor Torku")
plt.show()
```

<img width="571" alt="Ekran Resmi 2024-04-15 14 39 59" src="https://github.com/buzzi0/Task_Examples/assets/103946477/9a53fa94-c882-4556-9390-8b5c75afab7b">


```
sns.lineplot(x="hour", y="pCut_Motor_Torque", data=df, hue="anomaly")
plt.xlabel("Hour")
plt.ylabel("Motor Torku")
plt.title("Saatlere göre Motor torq değişimiyle anomali olup olmadığı tespiti")
plt.show()
```

<img width="621" alt="Ekran Resmi 2024-04-15 14 40 42" src="https://github.com/buzzi0/Task_Examples/assets/103946477/ed1d029a-bc63-4ee4-a130-69f3b9dda8e2">


```
sns.lineplot(
    x="Index",
    y="hour",
    hue="anomaly",
    data=df,
    
)
plt.title("Indexlerin Çalışma Saatlerine Göre Anomali Dağılımı")
plt.xlabel("Index")
plt.ylabel("Saat")
plt.show()
```

<img width="601" alt="Ekran Resmi 2024-04-15 14 41 23" src="https://github.com/buzzi0/Task_Examples/assets/103946477/ba9f2e2f-b6fa-4d66-a296-1beb6ff86486">


