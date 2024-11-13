import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree


# La donnée contient la liste de voiture allemande
cars = pd.read_csv('cars.csv', sep=",")


cars_data = cars[['make', 'mileage', 'price', 'year']]

X = cars_data.iloc[:,1:].values
Y = cars_data.iloc[:,0].values

print(X)
print(Y)

print('Labels:', cars['make'].unique())


from sklearn import tree

# Create and train the Decision Tree Classifier

# Import matplotlib for visualization
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Ensure class_names are strings
class_names = cars_data['make'].unique().astype(str)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X, Y)

plt.figure(figsize=(10, 10))
plot_tree(clf, feature_names=cars_data.columns, class_names=class_names, filled=True, rounded=True)

plt.show()

