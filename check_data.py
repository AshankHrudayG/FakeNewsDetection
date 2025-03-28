import pandas as pd

df = pd.read_csv("news.csv")
print(df.head())  # Show first few rows
print(df['label'].value_counts())  # Show count of Real and Fake news
