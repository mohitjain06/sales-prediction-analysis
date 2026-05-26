import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math  
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("train.csv")
df.info()
df.describe()
print(df.head())
Total_sales  = df["Sales"].sum()
print(Total_sales)
top_product = df.groupby("Product Name")["Sales"].sum()
print(top_product)
top_product.plot(kind="bar")
plt.show()
df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst= True)
print(df["Order Date"].head())
monthly = df.groupby(df["Order Date"].dt.month)["Sales"].sum()
monthly.plot(kind="bar")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
city_sales = df.groupby("City")["Sales"].sum()
city_sales.plot(kind="bar")
plt.title("City Wise Sales")
plt.xlabel("City")
plt.ylabel("Sales")
plt.xticks(rotation = 45)
plt.show()
category_sales = df.groupby("Category")["Sales"].sum()
category_sales.plot(kind="pie",autopct="%1.1f%%")
plt.title("Category Wise Sales")
plt.ylabel("")
plt.show()
top_product1 = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(5)
print(top_product1)
top_product1.plot(kind="bar")
plt.title("Top 5 product")
plt.show()
daily_sales = df.groupby("Order Date")["Sales"].sum()
print("Best Day:", daily_sales.idxmax())
print("Max Sales:", daily_sales.max())
daily_sales.plot(kind="line")
plt.title("Daily Sales Trend")
plt.show()

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst= True)
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Day"] = df["Order Date"].dt.day

df["Ship Date"]= pd.to_datetime(df["Ship Date"],dayfirst=True)
df["Shipping Time"] = (df["Ship Date"] - df["Order Date"]).dt.days


print(df[["Order Date", "Ship Date", "Shipping Time"]].head())
X = df[[
    "Month", "Year", 
    "Shipping Time",
    "Category", "Sub-Category",
    "Region", "Segment", "City", "Ship Mode"
]]

y = df["Sales"]

y = df["Sales"]
X = pd.get_dummies(X, drop_first=True)
y = np.log1p(df["Sales"])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr =LinearRegression()
rf = RandomForestRegressor()
xg = XGBRegressor()
lr.fit(X_train,y_train)
rf.fit(X_train,y_train)
xg.fit(X_train,y_train)

print("Linear:", lr.score(X_test,y_test))
print("Random Forest:", rf.score(X_test,y_test))
print("XGBoost:", xg.score(X_test,y_test))

for name,model in (["LR",lr],["RF",rf],["XG",xg]):
    pred = model.predict(X_test)
    print(name,"MAE:",mean_absolute_error(y_test,pred))

plt.bar(X.columns, xg.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()




