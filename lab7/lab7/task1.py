import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")
print(df.shape)
print(df.head())
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(3))

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print((df[cols] == 0).sum())
df[cols] = df[cols].replace(0, np.nan)
print(df.isna().sum())

f = ["Glucose", "BMI"]
x = df[f]
y = df["Outcome"].values

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

med = xtr.median()
xtr = xtr.fillna(med)
xte = xte.fillna(med)
print(med)

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

print(xtr.shape, xte.shape)
print(np.bincount(ytr), np.bincount(yte))
print(xtr.mean(axis=0).round(3), xtr.std(axis=0).round(3))
print(xte.mean(axis=0).round(3), xte.std(axis=0).round(3))

np.savez("data.npz", xtr=xtr, xte=xte, ytr=ytr, yte=yte)
