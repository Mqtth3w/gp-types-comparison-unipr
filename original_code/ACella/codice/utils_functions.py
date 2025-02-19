import random
import re
from deap import tools
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from nltk.tree import ParentedTree
from sklearn.model_selection import cross_val_score

def convolution(func, data, kernelsize,cnt):
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
def training_RF(X_data, X_labels, Y_data, Y_labels):
    try:
    # Convertire le etichette di classe in un formato adatto per la classificazione multi-classe
        le = LabelEncoder()
        le.fit(X_labels)
        X_labels_multi = le.transform(X_labels)
        Y_labels_multi = le.transform(Y_labels)

        seed = 40
        # salvare lo stato attuale del generatore di numeri casuali di numpy
        rng_state = np.random.get_state()

        # Creare il modello Random Forest per la classificazione multi-classe con lo stesso seed
        rf = RandomForestClassifier(n_estimators=50, random_state=seed)

        # ripristinare lo stato del generatore di numeri casuali di numpy
        np.random.set_state(rng_state)

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

    except Exception as e:
        # In caso di eccezione, impostiamo la fitness a zero
        mean_f1 = 0
        y_predictions = None
        rf = None
        Y_labels_multi = None
        
    return mean_f1 , Y_labels_multi, y_predictions, rf





def extraction(individuo):
    individuo=str(individuo).replace(" ","")
   
    # Regex per i sottomoduli di profondità 1
    regex_depth1 = r'(?:add|sub|neg|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|-?\d+,-?\d+|[-A-Za-z0-9_]+,-?\d+|[-A-Za-z0-9_]+,[A-Za-z0-9_]+)\)'
    regex_execTree_depth1 = r'execTree\d+\((?:-?\d+|ARG\d+|[A-Za-z0-9_]+)(?:,-?\d+|,ARG\d+|,[A-Za-z0-9_]+){0,3}\)'

    # Regex per i sottomoduli di profondità 2
    regex_depth2 = r'(?:sub|add|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\)'
    regex_neg2=r'(?:neg)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\)'
    regex_depth2_exec = r'(?:execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1+r'|'+regex_depth1 + r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\)'

    # Trova i sottomoduli di profondità 1
    sottomoduli_depth1 = re.findall(regex_depth1, individuo)
    sottomoduli_depth1.extend(re.findall(regex_execTree_depth1, individuo))
    # Trova i sottomoduli di profondità 2
    sottomoduli_depth2 = re.findall(regex_depth2_exec, individuo)
    sottomoduli_depth2.extend(re.findall(regex_depth2, individuo))
    sottomoduli_depth2.extend(re.findall(regex_neg2, individuo))
    for module in sottomoduli_depth1:
        if module in sottomoduli_depth2:
            sottomoduli_depth2.remove(module)
    return sottomoduli_depth1,sottomoduli_depth2


#resistuisce i moduli/sottoalberi della popolazione
def get_modules(pop):
    my_dict1={}
    my_dict2={}
    for p in pop:
        module_depth1,module_depth2=extraction(str(p))  # funzione che estrae i moduli da un albero
        for m in module_depth1:
            if m not in my_dict1:
                my_dict1[m] = [1,p.fitness.values[0]]
            else:
                my_dict1[m][0] += 1
                my_dict1[m][1] += p.fitness.values[0]
        for m in module_depth2:
            if m not in my_dict2:
                my_dict2[m] = [1,p.fitness.values[0]]
            else:
                my_dict2[m][0] += 1
                my_dict2[m][1] += p.fitness.values[0]

    return my_dict1,my_dict2 

def get_modules_individual(individual):
    modules=[]
    module_depth1,module_depth2=extraction(individual)  # funzione che estrae i moduli da un albero
    modules.extend(module_depth1)
    modules.extend(module_depth2)
    return modules


def depth(stringa):
    stringa=str(stringa).replace(" ","")
    # Regex per i sottomoduli di profondità 1
    regex_depth1 = r'(?:add|sub|neg|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|-?\d+,-?\d+|[-A-Za-z0-9_]+,-?\d+|[-A-Za-z0-9_]+,[A-Za-z0-9_]+)\)'
    regex_execTree_depth1 = r'execTree\d+\((?:-?\d+|ARG\d+|[A-Za-z0-9_]+)(?:,-?\d+|,ARG\d+|,[A-Za-z0-9_]+){0,3}\)'

    # Regex per i sottomoduli di profondità 2
    regex_depth2 = r'(?:sub|add|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\)'
    regex_neg2=r'(?:neg)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_depth1 +r'|'+regex_execTree_depth1+ r')\)'
    regex_depth2_exec = r'(?:execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1+r'|'+regex_depth1 + r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|' + regex_execTree_depth1 +r'|'+regex_depth1+ r')\)'

    # Controlla se la stringa corrisponde alla regex di profondità 1
    if (re.match(regex_depth1, stringa) or re.match(regex_execTree_depth1, stringa)):
        return 1
    # Controlla se la stringa corrisponde alla regex di profondità 2
    if ((re.match(regex_depth2, stringa)) or (re.match(regex_neg2, stringa)) or (re.match(regex_depth2_exec, stringa))) :
        return 2
    # Se la stringa non corrisponde a nessuna delle regex, restituisce None
    return None

#visualizza i moduli che si presentano con una frequenza maggiore di 5
def view_hist1(module_freq):
    plt.subplots(figsize=(8,6))  # imposta la dimensione del grafico
    # crea una lista delle chiavi del dizionario
    keys = list(module_freq.keys())
    # crea una lista dei primi valori delle liste associate alle chiavi del dizionario
    values = [module_freq[key][0] for key in keys]
    plt.bar(keys,values)
    plt.xlabel('Moduli')
    plt.xticks(fontsize=8)
    plt.xticks(rotation=90)  
    plt.ylabel('Frequenza')
    plt.title('Istogramma dei moduli di profondità 1')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.938, bottom=0.195)
    plt.show()
  
def view_hist2(module_freq):
    plt.subplots(figsize=(8,6)) # imposta la dimensione del grafico
    # crea una lista delle chiavi del dizionario
    keys = list(module_freq.keys())
    # crea una lista dei primi valori delle liste associate alle chiavi del dizionario
    values = [module_freq[key][0] for key in keys]
    plt.bar(keys,values)
    plt.xlabel('Moduli')
    plt.xticks(fontsize=8)
    plt.xticks(rotation=90) 
    plt.ylabel('Frequenza')
    plt.title('Istogramma dei moduli di profondità 2')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.938, bottom=0.195)
    plt.show()


def view_hist_fitness_freq(modules_freq_fitness):
    # Creazione del grafico
    fig, ax2 = plt.subplots()
    keys = list(modules_freq_fitness.keys())
    values_freq=[]
    values_fitness=[]
    # crea una lista dei primi valori delle liste associate alle chiavi del dizionario
    for key in keys:
        values_freq.append(modules_freq_fitness[key][0])
        values_fitness.append(modules_freq_fitness[key][1])

    ax2.bar(keys, values_freq, color="blue", label="Frequenza")
    ax1 = ax2.twinx()
    ax1.scatter(keys, values_fitness, color="red", label="Fitness normalizzate")
    
    # Personalizzazione del grafico
    ax1.set_xlabel("Moduli")
    ax1.set_ylabel("Valori di fitness normalizzata")
    ax2.set_ylabel("Valori di frequenza")
    fig.legend()
   
    # Visualizzazione del grafico
    plt.show()



def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple_elit(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best_ind=None
    # Begin the generational process
    for gen in range(1, ngen + 1):

        if best_ind is not None:
            random_index = random.randint(0, len(population) - 1)
            population[random_index] = best_ind

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        best_ind=halloffame[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook