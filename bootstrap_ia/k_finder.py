import MNISTAnalyser
import numpy as np
import matplotlib.pyplot as plt



from MNISTAnalyser import *
from knn_algo import KnnOperateur
import time
from datetime import datetime

analyser = MNISTAnalyser()

data_train, data_test = analyser.traitement()
label_train = analyser.train_labels
label_test = analyser.test_labels

#SUBTEST

label_test = label_test[:1000]
data_test = data_test[:1000]

label_train = label_train[:5000]
data_train = data_train[:5000]

accuracies = []

start_time = time.time()
start_datetime = datetime.now()
print(f"Début : {start_datetime.strftime('%H:%M:%S')}")

for k in range(1,21):
    print(f"Testing k={k}...")
    knn = KnnOperateur(k)
    knn.fit(data_train, label_train)
    predictions = knn.predict(data_test)
    accuracy = knn.accuracy(predictions, label_test)
    accuracies.append(accuracy)


end_time = time.time()
end_datetime = datetime.now()
elapsed_time = end_time - start_time

print(f"Fin : {end_datetime.strftime('%H:%M:%S')}")
print(f"Temps total : {elapsed_time:.2f} secondes ({elapsed_time/60:.2f} minutes)")

plt.plot(range(1,21),accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xlim(0.905, 1)
plt.show()

