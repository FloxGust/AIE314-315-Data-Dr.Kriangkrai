import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('supermarket_sales - Sheet1.csv')
df_selected = data[['Gender', 'Branch', 'Product line', 'Quantity', 'Total', 'Unit price']]

print("________________________________Check Data__________________________________") 
print("จำนวนของค่า null ในแต่ละ attribute:\n", data.isnull().sum(), "\n")
print(df_selected['Gender'].value_counts(), "\n")
print(df_selected['Branch'].value_counts(), "\n")

print("___________________________________________________ Exercies 4 ___________________________________________________")
for col in ['Quantity', 'Total']:
    print(f"\n{col}: \nMax = {df_selected[col].max()}\nMin = {df_selected[col].min()}\nAverage = {df_selected[col].mean():.2f}\n")

plt.figure(figsize=(12, 8))
for i, (col, color) in enumerate([('Gender', 'skyblue'), ('Branch', 'lightgreen'), ('Quantity', 'lightcoral'), ('Total', 'orange')], 1):
    plt.subplot(2, 2, i)
    if col in ['Gender', 'Branch']:
        df_selected[col].value_counts().plot(kind='bar', color=color)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.xticks(rotation=0)  
    else:
        plt.hist(df_selected[col], bins=10, color=color, edgecolor='black')
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
    plt.ylabel('Count' if col in ['Gender', 'Branch'] else 'Frequency')
plt.tight_layout()
plt.show()

print("___________________________________________________ Exercies 5 ___________________________________________________")
quantity_by_product = data.groupby('Product line')['Quantity'].sum()
quantity_percentage = (quantity_by_product / data['Quantity'].sum()) * 100
quantity_percentage = quantity_percentage.round(2) 
print("\n% ของจำนวนสินค้าทั้งหมดแยกตามประเภทสินค้า:\n", quantity_percentage)

quantity_percentage.plot(kind='bar', color='yellow', edgecolor='black')
plt.title('Percentage of Total Quantity by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=-45)
plt.tight_layout()
plt.show()

print("___________________________________________________ Exercies 6 ___________________________________________________")
sample_data = df_selected.sample(frac=0.3, random_state=42)
sample_percentage = (sample_data.groupby('Product line')['Quantity'].sum() / sample_data['Quantity'].sum()) * 100
print(sample_percentage.round(2))

print("___________________________________________________ Exercies 7 ___________________________________________________")
data = pd.read_csv('supermarket_sales - Sheet1.csv')
df = data[['Gender', 'Branch', 'Product line', 'Quantity', 'Total', 'Unit price','Date']].copy()


scaler = MinMaxScaler()
df.loc[:, 'Normalized Unit price'] = scaler.fit_transform(df[['Unit price']])
df['Normalized Unit price'] = df['Normalized Unit price'].round(2)

print(df[['Unit price', 'Normalized Unit price']].head())


print("___________________________________________________ Exercies 8 ___________________________________________________")

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Quantity'].sum()
print(monthly_sales)

sample_data = df.head(1000)
sampled_data = sample_data.groupby('Product line', group_keys=False).apply(lambda x: x.sample(frac=0.3, random_state=42))
print(sampled_data)