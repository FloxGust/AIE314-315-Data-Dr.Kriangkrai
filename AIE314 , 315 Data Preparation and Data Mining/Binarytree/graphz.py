# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import export_graphviz
import graphviz

# Step 2: Create the dataset from the table
data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocess the categorical data using LabelEncoder
label_encoder = preprocessing.LabelEncoder()

df['Outlook'] = label_encoder.fit_transform(df['Outlook'])
df['Temp'] = label_encoder.fit_transform(df['Temp'])
df['Humidity'] = label_encoder.fit_transform(df['Humidity'])
df['Wind'] = label_encoder.fit_transform(df['Wind'])
df['Play'] = label_encoder.fit_transform(df['Play'])  # 'Yes' = 1, 'No' = 0

# Features and target
X = df[['Outlook', 'Temp', 'Humidity', 'Wind']]  # Features
y = df['Play']  # Target

# Step 4: Split the data (although small, we'll still split for training/testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Build the Decision Tree model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make predictions and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 7: Visualize the decision tree using Graphviz
# Export the tree into a Graphviz format
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=['Outlook', 'Temp', 'Humidity', 'Wind'],  
                           class_names=['No', 'Yes'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
# Generate the Graphviz tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Save as a .pdf file

# Display the tree in Jupyter notebook (optional)
graph.view()
