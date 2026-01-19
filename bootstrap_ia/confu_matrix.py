from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from MNISTAnalyser import MNISTAnalyser
from knn_algo import KnnOperateur

analyser = MNISTAnalyser()
data_train, data_test = analyser.traitement()
label_train = analyser.train_labels
label_test = analyser.test_labels

#K trouvé
k_opti = 4

label_test_subset = label_test[:2000]
data_test_subset = data_test[:2000]

label_train_subset = label_train[:10000]
data_train_subset = data_train[:10000]

knn = KnnOperateur(k_opti)
knn.fit(data_train_subset, label_train_subset)
predictions = knn.predict(data_test_subset)

#Confusion matrix
cm = confusion_matrix(label_test_subset, predictions)

# Calcul du taux d'erreur par digit
total_per_digit = cm.sum(axis=1)  # somme par ligne
correct_per_digit = np.diag(cm)   # diagonal
error_rates = 1 - (correct_per_digit / total_per_digit)


#Visualisation

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predictions')
plt.ylabel('True')
plt.title(f'Confusion matrix (k={k_opti}) ')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(10), error_rates * 100)
plt.xlabel('Digit')
plt.ylabel('Error Rate (%)')
plt.title(f'Taux d\'erreur par digit (k={k_opti})')
plt.show()