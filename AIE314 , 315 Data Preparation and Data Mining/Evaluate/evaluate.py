import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_excel('customer_churn.xlsx', sheet_name='test_churn')
print('________________________________________________________ ข้อ 1 __________________________________________')

TP = len(data[(data['Churn'] == 'churn') & (data['prediction(Churn)'] == 'churn')])
FP = len(data[(data['Churn'] == 'loyal') & (data['prediction(Churn)'] == 'churn')])
FN = len(data[(data['Churn'] == 'churn') & (data['prediction(Churn)'] == 'loyal')])
TN = len(data[(data['Churn'] == 'loyal') & (data['prediction(Churn)'] == 'loyal')])

print(f'True Positive (TP): {TP}')
print(f'False Positive (FP): {FP}')
print(f'True Negative (TN): {TN}')
print(f'False Negative (FN): {FN}')
print()


print('________________________________________________________ ข้อ 2 __________________________________________')

accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
FNR = FN / (FN + TP) if (FN + TP) > 0 else 0


print('Accuracy:',"%.4f" % accuracy)
print('Precision:',"%.4f" % precision)
print('Recall:',"%.4f" % recall)
print('F1 Score:',"%.4f" % F1_score)
print('False Positive Rate:',"%.4f" % FPR)
print('False Negative Rate:',"%.4f" % FNR)

print('________________________________________________________ ข้อ 3 __________________________________________')

y_true = data['Churn'].apply(lambda x: 1 if x == 'churn' else 0)
y_pred = data['prediction(Churn)'].apply(lambda x: 1 if x == 'churn' else 0)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['loyal', 'churn'])
disp.plot()
plt.show()


