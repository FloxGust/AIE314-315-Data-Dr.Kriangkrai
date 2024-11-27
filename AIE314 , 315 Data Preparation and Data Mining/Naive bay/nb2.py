# from ucimlrepo import fetch_ucirepo
# import pandas as pd

# # Fetch the dataset
# mushroom = fetch_ucirepo(id=73)

# # Extract features and targets
# X = mushroom.data.features
# y = mushroom.data.targets

# # # Display basic information about the dataset
# # print("Metadata:")
# # print(mushroom.metadata)

# # print("\nVariable Information:")
# # print(mushroom.variables)

# # # Display the first few rows of the dataset
# # print("\nFirst few rows of the feature data:")
# # print(X.head())

# # print("\nFirst few rows of the target data:")
# # print(y.head())

# # # Display basic statistics
# # print("\nSummary statistics of feature data:")
# # print(X.describe(include='all'))



# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import numpy as np

# # Convert categorical features to numerical (if needed)
# X_encoded = pd.get_dummies(X)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Initialize classifiers
# classifiers = {
#     'Decision Tree': DecisionTreeClassifier(),
#     'k-NN': KNeighborsClassifier(),
#     'Naive Bayes': GaussianNB()
# }

# # Store the results
# results = {}

# for name, clf in classifiers.items():
#     scores = cross_val_score(clf, X_encoded, y, cv=10, scoring='accuracy')
#     accuracy = np.mean(scores)
    
#     # Fit the classifier and calculate precision and recall
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
    
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
    
#     results[name] = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall
#     }

# # Display the results
# for name, metrics in results.items():
#     print(f"\n{name}:")
#     print(f"Accuracy: {metrics['accuracy']:.4f}")
#     print(f"Precision: {metrics['precision']:.4f}")
#     print(f"Recall: {metrics['recall']:.4f}")




# import matplotlib.pyplot as plt

# # Data for plotting
# labels = list(results.keys())
# accuracy_scores = [metrics['accuracy'] for metrics in results.values()]
# precision_scores = [metrics['precision'] for metrics in results.values()]
# recall_scores = [metrics['recall'] for metrics in results.values()]

# x = np.arange(len(labels))  # the label locations
# width = 0.25  # the width of the bars

# fig, ax = plt.subplots()
# bars1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy')
# bars2 = ax.bar(x, precision_scores, width, label='Precision')
# bars3 = ax.bar(x + width, recall_scores, width, label='Recall')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of Classifier Metrics')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # Show the bar graph
# plt.show()



from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Fetch the dataset
mushroom = fetch_ucirepo(id=73)

# Extract features and targets
X = mushroom.data.features
y = mushroom.data.targets.values.ravel()  # Flatten y to 1D array

# Convert categorical features to numerical (if needed)
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Store the results
results = {}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_encoded, y, cv=10, scoring='accuracy')
    accuracy = np.mean(scores)
    
    # Fit the classifier and calculate precision and recall
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

# Display the results
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

import matplotlib.pyplot as plt

# Data for plotting
labels = list(results.keys())
accuracy_scores = [metrics['accuracy'] for metrics in results.values()]
precision_scores = [metrics['precision'] for metrics in results.values()]
recall_scores = [metrics['recall'] for metrics in results.values()]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

# Adjusting the bar placements
bars1 = ax.bar(x, accuracy_scores, width, label='Accuracy')
bars2 = ax.bar(x + width, precision_scores, width, label='Precision')
bars3 = ax.bar(x + 2 * width, recall_scores, width, label='Recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Comparison of Classifier Metrics')
ax.set_xticks(x + width)  # Centering labels with respect to the bars
ax.set_xticklabels(labels)
ax.legend()

# Show the bar graph
plt.show()

