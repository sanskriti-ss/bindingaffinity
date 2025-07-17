import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/home/karen/Projects/FAST/data/refined_3d_out.csv')

# Create histogram of binding affinity (label column)
plt.figure(figsize=(10, 6))
plt.hist(df['label'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Binding Affinity (pKd/pKi)')
plt.ylabel('Frequency')
plt.title('Distribution of Binding Affinity Values')
plt.grid(True, alpha=0.3)

# Add some statistics to the plot
mean_affinity = df['label'].mean()
std_affinity = df['label'].std()
plt.axvline(mean_affinity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_affinity:.2f}')
plt.axvline(mean_affinity + std_affinity, color='orange', linestyle='--', alpha=0.7, label=f'Mean + 1σ: {mean_affinity + std_affinity:.2f}')
plt.axvline(mean_affinity - std_affinity, color='orange', linestyle='--', alpha=0.7, label=f'Mean - 1σ: {mean_affinity - std_affinity:.2f}')

plt.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print(f"Summary Statistics for Binding Affinity:")
print(f"Mean: {mean_affinity:.2f}")
print(f"Std: {std_affinity:.2f}")
print(f"Min: {df['label'].min():.2f}")
print(f"Max: {df['label'].max():.2f}")
print(f"Median: {df['label'].median():.2f}")
print(f"Total samples: {len(df)}")