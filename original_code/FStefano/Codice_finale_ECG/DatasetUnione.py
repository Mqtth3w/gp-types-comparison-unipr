
import pandas as pd

dataset1 = pd.read_csv('Progetto Maffoni\output_abn.csv')
dataset2 = pd.read_csv('Progetto Maffoni\output_norm.csv')

appended_dataset = pd.concat([dataset1, dataset2], ignore_index=True)

missing_values = appended_dataset.isnull().sum()
#print("Conteggio dei valori mancanti per colonna:\n", missing_values)

appended_dataset.to_csv('Dfinale.csv', index=False)