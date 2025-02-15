# initially developed by Aeranna Cella, reviewed by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
import pickle
import dill
from deap import gp
from utils import convolution, training_rf
from data_loader import load_dataset

# chose dataset
train_data, train_labels, data_val, labels_val, test_data, test_labels = load_dataset("_")

def eval(individual, data, labels, pset):
    clf = gp.PrimitiveTree(individual)
    func = gp.compile(clf, pset)
    new_train_set = convolution(func, train_data, KERNEL_SIZE)
    new_test_set = convolution(func, data, KERNEL_SIZE)
    f1_t = training_rf(new_train_set, train_labels, new_test_set, labels)
    return f1_t

# results file
with open("_.txt", "r") as r:
    r.readline() # skip header
    KERNEL_SIZE = int((r.readline().split(';'))[6])
    print(f"KERNEL_SIZE: {KERNEL_SIZE}")

# pset file with the same method and time of results file (for the modified gp also the run)
with open("_.pkl", "rb") as f:
    pset = dill.load(f)

# best individual with the same method and time of results file (for the modified gp also the run)
with open("_.pickle", "rb") as f:
    best_individual = pickle.load(f)

# arbitrary test
data = test_data[20:40]
labels = test_labels[20:40]

# test
f1 = eval(best_individual, data, labels, pset)
print(f1)