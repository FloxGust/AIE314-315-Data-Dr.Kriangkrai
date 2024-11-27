import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
 
# 2
df = pd.read_csv('supermarket_sales.csv')
missing_values = df.isnull().sum()
print(missing_values)
 
# 3
attributes = df[['Gender', 'Product line', 'Total', 'Quantity', 'Branch']]
print(attributes.dtypes)
 
# 4
stat = attributes.describe()
print(stat)
attributes['Gender'].value_counts().plot(kind='bar', color='deeppink')
plt.title('Distribution of Gender')
plt.xticks(rotation = 0)
plt.show()
 
attributes['Product line'].value_counts().plot(kind='bar', color='lime')
plt.title('Distribution of Product line')
plt.xticks(rotation= 35)
plt.show()
 
attributes['Branch'].value_counts().plot(kind='bar', color='deeppink')
plt.title('Distribution of Branch')
plt.xticks(rotation = 0)
plt.show()
 
# 5
quantity = df.groupby('Product line')['Quantity'].sum()
percentage = (quantity / quantity.sum()) * 100
percentage = percentage.round(2)  
print(percentage)
 
# 6
sample_data = df.sample(frac=0.3, random_state=42)
sample_quantity = sample_data.groupby('Product line')['Quantity'].sum()
sample_percentage = (sample_quantity / sample_quantity.sum()) * 100
sample_percentage = sample_percentage.round(2)  
print(sample_percentage)
 
# 7
scaler = MinMaxScaler()
df['Normalized Unit price'] = scaler.fit_transform(df[['Unit price']])
df['Normalized Unit price'] = df['Normalized Unit price'].round(2)  
print(df[['Unit price', 'Normalized Unit price']].head())
 
#Option
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Quantity'].sum()
print(monthly_sales)
 
 
sample_data = df.head(1000)
sampled_data = sample_data.groupby('Product line', group_keys=False).apply(lambda x: x.sample(frac=0.3, random_state=42))
print(sampled_data)
 