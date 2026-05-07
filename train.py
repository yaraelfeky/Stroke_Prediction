import pandas as pd
from model import train_model

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

train_model(df)

print("Model trained successfully!")