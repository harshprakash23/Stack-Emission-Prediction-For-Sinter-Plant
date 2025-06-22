import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Loading the data
df = pd.read_csv('INTERNSHIP_DATA.csv')

# Cleaning and converting CLOCK to datetime
df['CLOCK'] = df['CLOCK'].str.strip()  # Remove leading/trailing spaces
df['CLOCK'] = pd.to_datetime(df['CLOCK'], format='mixed', dayfirst=True, errors='coerce')

# Check for any parsing errors
if df['CLOCK'].isnull().any():
    print("Warning: Some CLOCK values could not be parsed. Invalid rows:")
    print(df[df['CLOCK'].isnull()])
    df = df.dropna(subset=['CLOCK'])  # Drop rows with invalid CLOCK

# Setting CLOCK as index
df.set_index('CLOCK', inplace=True)

# 1. Basic Inspection
print("Data Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 2. Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Distribution Plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('distributions.png')
plt.close()

# 4. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# 5. Time Series Plot of Output
plt.figure(figsize=(15, 5))
df['output'].plot()
plt.title('Stack Emission (Output) Over Time')
plt.xlabel('Time')
plt.ylabel('Output')
plt.savefig('output_timeseries.png')
plt.close()

# 6. Boxplots for Outlier Detection
plt.figure(figsize=(15, 10))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot for Outlier Detection')
plt.savefig('boxplots.png')
plt.close()

# 7. Scatter Plots for Key Correlated Features
high_corr = df.corr()['output'].abs().sort_values(ascending=False).index[1:4]  # Top 3 correlated features
for col in high_corr:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[col], df['output'], alpha=0.5)
    plt.title(f'{col} vs Output')
    plt.xlabel(col)
    plt.ylabel('Output')
    plt.savefig(f'scatter_{col}_vs_output.png')
    plt.close()

print("EDA plots saved successfully.")