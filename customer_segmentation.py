import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# reading the dataset
df = pd.read_csv("data/Mall_Customers.csv")
print("✅ Data loaded successfully!")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5,random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2'
)
plt.title("Customer Segmentation")
plt.show()

os.makedirs("output", exist_ok=True)
df.to_csv("output/customer_segments.csv",index=False)

# EDA Visualizations
sns.histplot(df['Age'], bins=20,kde=True,color="skyblue")
plt.title("Age Distribution")
plt.show()

sns.histplot(df['Annual Income (k$)'],bins=20,kde=True,color="orange")
plt.title("Annual Income Distribution")
plt.show()

sns.histplot(df['Spending Score (1-100)'], bins=20,kde=True,color="green")
plt.title("Spending Score Distribution")
plt.show()

sns.countplot(x='Gender', data=df, palette="Set2")
plt.title("Gender Count")
plt.show()

sns.heatmap(df[['Age','Annual Income (k$)','Spending Score (1-100)']].corr(),annot=True, cmap="coolwarm")      
plt.title("Correlation Heatmap")
plt.show()

