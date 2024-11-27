import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
print(df, f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")

def print_numeric_stats(column):
    print(f"\nColumn: {column.name}\nMax: {column.max()},\n Min: {column.min()},\n Avg: {column.mean():.2f}\n")
    column.hist(bins=32, color='red', edgecolor='yellow', figsize=(8, 6))
    plt.title(f"Histogram of {column.name}")
    plt.xlabel(column.name)
    plt.ylabel("Frequency")
    plt.show()

def category(column):
    category_counts = column.value_counts()
    print(f"\nมีข้อมูล category ทั้งหมด {len(category_counts)} ค่า")
    for category, count in category_counts.items():
        print(f"ประเภท {category}: {count} แถว")

for col in df.drop(columns='CustomerID'):
    if pd.api.types.is_numeric_dtype(df[col]):
        print_numeric_stats(df[col])
    elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
        category(df[col])