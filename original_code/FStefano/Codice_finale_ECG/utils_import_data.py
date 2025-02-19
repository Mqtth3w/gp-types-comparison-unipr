import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  

def import_from_file(file_path: str) -> tuple:
    df = pd.read_csv(file_path)
    labels = df.iloc[:, 0].tolist()  # Prima colonna come etichette
    data = df.iloc[:, 1:-1].values  # Colonne centrali come dati, usa .values per ottenere ndarray
    targets = df.iloc[:, -1].tolist()  # Ultima colonna come target
    return labels, data, targets

def shuffle_in_unison(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def get_dataset(dataset):
    labels, data, targets = import_from_file(dataset)
    labels, data, targets = shuffle_in_unison(np.array(labels), np.array(data), np.array(targets))
    
    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, targets, test_size=0.2, random_state=42)
    train_data, data_val, train_labels, labels_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    return train_data, train_labels, test_data, test_labels, data_val, labels_val