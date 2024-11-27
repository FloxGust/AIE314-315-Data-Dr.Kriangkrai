import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

df = pd.read_csv('drug200.csv')


label_encoder = preprocessing.LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']] 
y = df['Drug']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

clf_tuned = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, random_state=42)
clf_tuned.fit(X_train, y_train)

y_pred_tuned = clf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Accuracy: {accuracy_tuned:.2f}")



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plot_tree(clf_tuned, filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], class_names=clf_tuned.classes_)
plt.show()




