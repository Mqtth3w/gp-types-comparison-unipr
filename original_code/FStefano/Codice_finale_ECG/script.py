import operator
import pickle
import random
import dill
from deap import creator, base, tools, gp
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils_functions import convolution, training_RF
from utils_import_data import get_dataset


train_data, train_labels, test_data, test_labels, data_val, labels_val = get_dataset("dataset\digits_dataset.csv")

def convolution(func, data, kernelsize):
    data = np.array(data)
    #print("training set:",data)
    new_data = []
    for image in data:
        #print("image i:",image)
        mapping = {}
        new_image = []
        for i in range(len(image) - kernelsize + 1):
            mapping.clear()
            for j in range(0,kernelsize):
                mapping["ARG" + str(j)] = image[j+i]
            
            number = func(**mapping)
            new_image.append(number)
            #print("new image",new_image)
        new_data.append(new_image)
    #print("new training set:",new_data)
    return np.array(new_data)

#addestramento/valutazione random forest
def training_RF(X_data, X_labels, Y_data, Y_labels,num_nodes):
    
   # Convertire le etichette di classe in un formato adatto per la classificazione multi-classe
    le = LabelEncoder()
    le.fit(X_labels)
    X_labels_multi = le.transform(X_labels)
    Y_labels_multi = le.transform(Y_labels)

    # Creare il modello Random Forest per la classificazione multi-classe
    rf = RandomForestClassifier(n_estimators=50)
    # Addestrare il modello
    rf.fit(X_data, X_labels_multi)
    # Valutare il modello usando il validation set per generalizzare meglio
    y_predictions = rf.predict(Y_data)

    # Calcolare la media degli F1-score per ogni classe
    f1_per_class = []
    for i in range(len(le.classes_)):
        #print(i)
        y_true_i = [int(label == i) for label in Y_labels_multi]
        y_pred_i = [int(pred == i) for pred in y_predictions]
        f1_i = f1_score(y_true_i, y_pred_i, zero_division=0)
        f1_per_class.append(f1_i)
    mean_f1 = sum(f1_per_class) / len(f1_per_class)

    return mean_f1 

def eval(individual,data,labels, pset):
    clf = gp.PrimitiveTree(individual)
    func = gp.compile(clf, pset) # che pset uso? ci sono execTree dovrò salvarmi anche quello?
    num_nodes=individual.height
    new_train_set=convolution(func,train_data, KERNEL_SIZE)
    new_test_set=convolution(func,data,KERNEL_SIZE)
    f1_t=training_RF(new_train_set,train_labels, new_test_set, labels, num_nodes)
    return f1_t

with open("risultati_digits_905\efimcost_2023-05-09_13-37-36.txt", "r") as r:
    const=int(r.readline())
    KERNEL_SIZE=int(r.readline())
    
pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
pset.addEphemeralConstant(f"rand101_{const}", lambda: random.randint(-1, 1))

with open("risultati_digits_905\pset_2023-05-09_13-37-36.pkl", "rb") as f:
    pset = dill.load(f)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


with open("risultati_digits_905\\best_individual_2023-05-09_13-37-36.pickle", "rb") as f:
    best_individual = pickle.load(f)

best_individual = creator.Individual(best_individual)
data=test_data[20:40]
labels=test_labels[20:40]
f1=eval(best_individual,data,labels,pset)
print(f1)