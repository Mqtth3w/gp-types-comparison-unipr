# initially developed by Aeranna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import pandas as pd
from sklearn.model_selection import train_test_split

def import_from_file(file_path: str):
    """Load data from a CSV file using pandas."""
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values  # all except first col
    labels = df.iloc[:, 0].values  # first col
    return data, labels

def load_dataset(file_path):
    """Load and split dataset in training, validation, and test set."""
    data, labels = import_from_file(file_path)
    
    # Split into training and temp (validation + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.4, random_state=42, shuffle=True
    )
    
    # Split temp into validation and test
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, shuffle=True
    )
    
    # training 60%, validation 20%, test 20%
    return train_data, train_labels, val_data, val_labels, test_data, test_labels