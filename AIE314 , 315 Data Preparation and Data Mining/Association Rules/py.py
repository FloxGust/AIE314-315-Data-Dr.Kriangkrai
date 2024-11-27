import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
 
data = pd.read_csv('market_basket_dataset.csv')
df = pd.DataFrame(data, columns=['BillNo', 'Itemname'])
basket = df.groupby('BillNo')['Itemname'].apply(list)
 
te = TransactionEncoder()
te_data = te.fit(basket).transform(basket)
df_te = pd.DataFrame(te_data, columns=te.columns_)
 
frequent_itemsets = fpgrowth(df_te, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
 
print("FP-Growth algorithm :")
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])