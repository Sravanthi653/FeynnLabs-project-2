# Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess
df = pd.read_csv("household_appliances_sample.csv")
df['House_Type'] = LabelEncoder().fit_transform(df['House_Type'])
df['Urban_Rural'] = LabelEncoder().fit_transform(df['Urban_Rural'])
df = df.drop(columns=['ID'])
X_scaled = StandardScaler().fit_transform(df)