import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/fraud.csv")

# Display dataset
print(df.head())

# Dataset information
print("Dataset Shape:", df.shape)
print(df.info())

# Check missing values
print(df.isnull().sum())

# Count fraud and genuine transactions
print(df["isFraud"].value_counts())

# Plot fraud distribution
plt.bar([0, 1], df["isFraud"].value_counts().sort_index())

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.savefig("fraud_distribution.png")
plt.show()