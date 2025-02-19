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


"""
Queste funzioni vengono utilizzate per la valutazione degli individui e 
per l'addestramento e la valutazione di un modello di classificazione RF."""


def convolution(func, data, kernelsize, cnt):
    data = np.array(data)
    new_data = np.empty((len(data), len(data[0]) - kernelsize + 1))
    
    for idx, image in enumerate(data):
        new_image = [
            func(**{f"ARG{j}": image[i + j] for j in range(kernelsize)})
            for i in range(len(image) - kernelsize + 1)
        ]
        new_data[idx] = new_image
    
    return new_data


#addestramento/valutazione random forest
def training_RF(X_data, X_labels, Y_data, Y_labels):
    """
    Addestro un modello di classificazione RF e valuto la sua performance. 
    Le label di classe vengono codificate con 'LabelEncoder', il modello viene addestrato e vengono generate le predizioni.
    Le performance del modello viene misurata con la media degli F1-score per ogni classe.
    In caso di errore, viene restituito un valore di fitness pari a 0.
    """
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


        f1_per_class = [
            f1_score([int(label == i) for label in Y_labels_multi],
                     [int(pred == i) for pred in y_predictions],
                     zero_division=0)
            for i in range(len(le.classes_))
        ]
        mean_f1 = np.mean(f1_per_class)
        
        """         cm = confusion_matrix(Y_labels_multi, y_predictions)
                print("MATRICE DI CONFUSIONE:")
                print(cm)
                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                disp.plot()
                plt.show() """
    except Exception as e:
        mean_f1, y_predictions, rf, Y_labels_multi = 0, None, None, None
        
    return mean_f1, Y_labels_multi, y_predictions, rf




"""
Estraggo i moduli o sottoslberi di profondità 1 o 2 da un individuo rappresentato come una stringa 
Utlizzo espressioni regolari per identificare i moduli e restituire due liste separate
di moduli di profondità 1 e 2
"""
####LAVORO QUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
PER SOSTITUIRE LE REGEX CON LE LISTE DI ADICANZA, devo rappresentare gli alberi di espressioni 
matematiche come grafie. Ogni nodo del grafo rappresenta un' operazione o un numero e ogni arco
rappresenta un' operazione su due numeri o espressioni.


Quindi posso usare un dizionario di liste(?chiedi) per rappresentare gli alberi di espressioni


1)  Creo delle liste di adiacenza==> ogni operazione sarà un nodo e gli argomenti dell operazione saranno nodi collegati
    all' operazione stessa. Per esempio, add(1,2) avrà un nodo add collegato ai nodi '1' e '2'.


2)Rimuovo le regex==> rimppiazzo il parsing delle espressioni basato su regex con un parsing basato su liste di adiacenza
"""
################################################################################################################
def extraction(individuo):
    individuo = str(individuo).replace(" ", "")
    #print("Individuo : " + individuo)
    
    def parse_expression(expression):
        stack = []
        adj_list = {}
        current_node = None
        
        i = 0
        while i < len(expression):
            if expression[i].isalpha():
                j = i
                while j < len(expression) and (expression[j].isalpha() or expression[j].isdigit() or expression[j] == '_'):
                    j += 1
                node = expression[i:j]
                i = j
                if current_node is None:
                    current_node = node
                    adj_list[current_node] = []
                else:
                    stack.append(current_node)
                    current_node = node
                    adj_list[current_node] = []
            elif expression[i].isdigit() or expression[i] == '-':
                j = i
                while j < len(expression) and (expression[j].isdigit() or expression[j] == '-'):
                    j += 1
                node = expression[i:j]
                i = j
                adj_list[current_node].append(node)
            elif expression[i] == ',':
                i += 1
            elif expression[i] == ')':
                if stack:
                    parent_node = stack.pop()
                    adj_list[parent_node].append(current_node)
                    current_node = parent_node
                i += 1
            elif expression[i] == '(':
                i += 1
        #print("Adjacency list: ", adj_list)
        return adj_list
    
    return parse_expression(individuo)


# resistuisce i moduli/sottoalberi della popolazione
def get_modules(pop):
    my_dict1 = {}
    my_dict2 = {}
    
    for p in pop:
        adj_list = extraction(str(p))  # funzione che estrae i moduli da un albero
        
        for node in adj_list:
            children = adj_list[node]
            if len(children) == 1:  # modulo di profondità 1
                if node not in my_dict1:
                    my_dict1[node] = [1, p.fitness.values[0]]
                else:
                    my_dict1[node][0] += 1
                    my_dict1[node][1] += p.fitness.values[0]
            elif len(children) > 1:  # modulo di profondità 2
                if node not in my_dict2:
                    my_dict2[node] = [1, p.fitness.values[0]]
                else:
                    my_dict2[node][0] += 1
                    my_dict2[node][1] += p.fitness.values[0]


    return my_dict1, my_dict2


########################################################################


def get_modules_individual(individual):
    adj_list = extraction(individual)
    module_depth1 = []
    module_depth2 = []
    
    for node in adj_list:
        children = adj_list[node]
        if len(children) == 1:  # modulo di profondità 1
            module_depth1.append(node)
        elif len(children) > 1:  # modulo di profondità 2
            module_depth2.append(node)
    
    return module_depth1, module_depth2


def depth(individuo):
    adj_list = extraction(individuo)
    if not adj_list:
        print("Errore: adj_list è vuota.")
        return 0
    
    def max_depth(node):
        if node not in adj_list or not adj_list[node]:
            return 1
        else:
            return 1 + max(max_depth(child) for child in adj_list[node])
    
    return max(max_depth(node) for node in adj_list)



#visualizza i moduli che si presentano con una frequenza maggiore di 5
"""
Queste funzioni visulizzano istogrammi della frequenza dei moduli di profndità 1 e 2
i moduli sono rappresentati sulle ascisse e le loro frequenze sulle ordinate
"""
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


"""
Questa funzione visulizza un istogramma delle frequenze dei moduli e un grafico a dispersione 
dei valori di fitness normalizzati.
"""
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


"""
questa funzione applica crossover e mutazione alla popolazione. Utilizza le probabilità
di crossover ('cxpb') e mutazione ('mutpb') per determinare quali individui subiscono variazioni.
"""


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


"""
Questa funzione esegue l' algoritmo genetco con selezione elitista. Mantiene traccia delle milgiori soluzioni (hall of fame)
valuta la fitness degli individui e aggiorna la poplazione generazione dopo generazione.
I risultati vengono registrati in un logbook e possono essere visualizzati.


"""
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