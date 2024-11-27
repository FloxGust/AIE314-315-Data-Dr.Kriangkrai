import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

data = pd.read_csv('midtrem/HealthCareData2024.csv')
df = data[['AlertCategory','NetworkEventType', 'NetworkInteractionType', 'UserActivityLevel', 'SecurityRiskLevel', 'Classification']]

print(df.head(50))

print(df['AlertCategory'].value_counts(), "\n")

# quantity = data.groupby('AlertCategory')['NetworkInteractionType'].sum()
print(df['AlertCategory'].sum())




x_c = df['NetworkEventType', 'NetworkInteractionType', 'UserActivityLevel', 'SecurityRiskLevel', 'Classification']
y_c = df['AlertCategory']



# model = IsolationForest(contamination=0.05)
# df['AlertCategory'] = model.fit_predict()

anomaly_points = df['AlertCategory'] == 'Alert'

print(anomaly_points)

# plt.figure(figsize=(10, 6))
# plt.scatter(data[x_c], data[y_c], color='blue', label='Normal')
# plt.scatter(anomaly_points[x_c], anomaly_points[y_c], color='red', label='Anomaly')
# plt.title(f'Column {x_c}: Column {y_c}')
# plt.xlabel(x_c)
# plt.ylabel(y_c)
# plt.legend()
# plt.show()































































