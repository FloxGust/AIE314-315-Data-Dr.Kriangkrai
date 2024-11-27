import pandas as pd
import matplotlib.pyplot as plt

file_path = "supermarket_sales - Sheet1.csv"
data = pd.read_csv(file_path)


quantity_by_product = data.groupby('Product line')['Quantity'].sum()

total_quantity = data['Quantity'].sum()
quantity_percentage_by_product = (quantity_by_product / total_quantity) * 100

print("\n% ของจำนวนสินค้าทั้งหมดแยกตามประเภทสินค้า:")
print(quantity_percentage_by_product)

quantity_percentage_by_product.plot(kind='bar', color='yellow', edgecolor='black')
plt.title('Percentage of Total Quantity by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=-45)
plt.tight_layout()
plt.show()
