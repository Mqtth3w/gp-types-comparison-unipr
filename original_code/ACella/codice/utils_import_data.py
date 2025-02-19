import csv
import numpy as np
from sklearn.model_selection import train_test_split

def import_from_file(file_path: str) -> tuple:
    """
    I file CSV devono essere nel formato seguente:
        - la prima riga un intestazione con i nomi delle colonne
        - dalla seconda riga in poi la prima colonna è la label e le restati sono i dati che rappresentano un immagine caratterizzata dalla label in prima colonna
    """
    header = []
    labels = []
    data = []
    with open(file_path, "r") as file: #os.path.dirname(__file__) + "/" +
        file_reader = csv.reader(file)
        count = 0
        for line in file_reader:
            if count == 0:
                for value in line[1:]:
                    header.append(value)
            else:
                l = []
                for value in line[1:]:
                    l.append(float(value))
                data.append(l)
                labels.append(line[0])
            count += 1
    return header, labels, data

#mescola il posizionamento di due array non modificando la loro posizione reciproca
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


#ottengo training set, validation set e test set dal file CSV che viene importato già formattato
def get_dataset(dataset):
    x, labels, lines = import_from_file(dataset)
    labels, lines = shuffle_in_unison(np.array(labels), np.array(lines))
    labels = labels.tolist()
    lines = lines.tolist()
    train_data, test_data, train_labels, test_labels = train_test_split(lines , labels, test_size=0.2, random_state=42)
    # Divide il training set in training e validation set
    train_data, data_val, train_labels, labels_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    '''
    print(f"Training set contains {len(train_data)} total instances, {train_labels.count('0.0')} 0 and {train_labels.count('1.0')} 1")
    print(f"Test set contains {len(test_data)} total instances, {test_labels.count('0.0')} 0 and {test_labels.count('1.0')} 1")
    print(f"Validation set contains {len(data_val)} total instances, {labels_val.count('0.0')} 0 and {labels_val.count('1.0')} 1")
    '''
    return train_data, train_labels, test_data, test_labels, data_val, labels_val

