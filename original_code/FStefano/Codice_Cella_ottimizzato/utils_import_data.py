import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

#* PER RENDERE PIù EFFICIENTE IL CODICE SONO INTRODOTTE LE SEGUENTI MODIFICHE
    #*Utilizzare librerie più efficienti per l gestione dei dati CSV;
    #*Rimuovere il codice ridondante 
    #*Rendere il codice più leggibile 

def import_from_file(file_path: str) -> tuple:
    """
    I file CSV devono essere nel formato seguente:
        - la prima riga un intestazione con i nomi delle colonne
        - dalla seconda riga in poi la prima colonna è la label e le restati sono i dati che rappresentano un immagine caratterizzata dalla label in prima colonna
        
        Args: 
            file_path (str): percorso del file CSV
        Returns:
            tuple: una tupla con tre elementi, header, labels e data
                header (list): lista di stringhe con i nomi delle colonne
                labels (list): lista di stringhe con le label
                data (list): lista di liste con i dati
    """
    df = pd.read_csv(file_path)
    header = list(df.columns[1:])
    labels = df.iloc[:, 0].tolist()
    data = df.iloc[:, 1:].values.tolist()
    return header, labels, data

#mescola il posizionamento di due array non modificando la loro posizione reciproca
def shuffle_in_unison(a, b):
    """
    Mescola due array in modo sincrono, mantenendo la corrispondenza tra gli elementi degli array.
    
    Args:
        a (numpy.array): primo array da mescolare
        b (numpy.array): secondo array da mescolare
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


#ottengo training set, validation set e test set dal file CSV che viene importato già formattato
def get_dataset(dataset):
    """
    Args:
        dataset (str): percorso del file CSV
    Returns:
        tuple: una tupla con sei elementi: train_data, train_labels, test_data, test_labels, data_val, labels_val
            train_data (list): lista di liste con i dati di training
            train_labels (list): lista di stringhe con le label di training
            test_data (list): lista di liste con i dati di test
            test_labels (list): lista di stringhe con le label di test
            data_val (list): lista di liste con i dati di validazione
            labels_val (list): lista di stringhe con le label di validazione
    """
    # Importa i dati dal file 
    header, labels, data = import_from_file(dataset)
    
    # Mescola le etichette e i dati in modo sincrono
    labels, data = shuffle_in_unison(np.array(labels), np.array(data))
    
    # Divide i dati nei set di training, test e validazione
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_data, data_val, train_labels, labels_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
    return train_data, train_labels, test_data, test_labels, data_val, labels_val

