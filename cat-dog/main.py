import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.io import imread
CLASSES = {"cats" : 0, "dogs" : 1}
DATA_DIRECTORY = Path("dataset")
images, labels = [], []
for folder, label in CLASSES.items():
  for path in ((DATA_DIRECTORY / folder).glob("*.png")):
    current_image = imread(path)
    current_image = current_image.reshape(-1)
    images.append(current_image)
    labels.append(label)
images, labels = np.array(images), np.array(labels)
print(images.shape)
print(labels.shape)
x_train, x_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.25,
    stratify=labels)
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_train, y_train)
print(tree_clf.score(x_test, y_test))
from sklearn.svm import SVC
support_vectors_clf = SVC(kernel='rbf', C=1.0)
support_vectors_clf.fit(x_train, y_train)
print(support_vectors_clf.score(x_test, y_test))
