import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/placement.csv")

# Show first 5 rows
print(data.head())

# Describe dataset
print(data.describe())

# Placement distribution
sns.countplot(x="Placed", data=data)
plt.title("Placement Distribution")
plt.show()

# CGPA vs Placement
sns.boxplot(x="Placed", y="CGPA", data=data)
plt.title("CGPA vs Placement")
plt.show()

# Correlation heatmap
numeric_data = data.select_dtypes(include='number')  # only numeric columns
sns.heatmap(numeric_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()