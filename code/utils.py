# initially developed by Aeranna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
import re
import random
from deap import tools
from deap.algorithms import varAnd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def convolution(func, data, kernelsize):
    data = np.array(data)
    new_data = np.empty((len(data), len(data[0]) - kernelsize + 1))
    
    for idx, image in enumerate(data):
        new_image = [
            func(**{f"ARG{j}": image[i + j] for j in range(kernelsize)})
            for i in range(len(image) - kernelsize + 1)
        ]
        new_data[idx] = new_image
    
    return new_data

# random forest training/evaluation
def training_rf(X_data, X_labels, Y_data, Y_labels):
    try:
        le = LabelEncoder()
        all_labels = np.concatenate((X_labels, Y_labels))
        le.fit(all_labels)
        
        X_labels_multi = le.transform(X_labels)
        Y_labels_multi = le.transform(Y_labels)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_data, X_labels_multi)

        y_predictions = rf.predict(Y_data)
        mean_f1 = f1_score(Y_labels_multi, y_predictions, average='macro', zero_division=0)

    except Exception as e:
        print(f"Error (training_rf): {e}")
        return 0, None, None, None

    return mean_f1, Y_labels_multi, y_predictions, rf

def extraction_tree(individual): # O(n)
    individual = str(individual).replace(" ","")

    # regex for depth 1 submodules
    regex_depth1 = r'(?:add|sub|neg|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|-?\d+,-?\d+|[-A-Za-z0-9_]+,-?\d+|[-A-Za-z0-9_]+,[A-Za-z0-9_]+)\)'
    regex_execTree_depth1 = r'execTree\d+\((?:-?\d+|ARG\d+|[A-Za-z0-9_]+)(?:,-?\d+|,ARG\d+|,[A-Za-z0-9_]+){0,3}\)'

    # regex for depth 2 submodules
    regex_depth2 = r'(?:sub|add|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\)'
    regex_neg2 = r'(?:neg)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\)'
    regex_depth2_exec = r'(?:execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\)'

    # find depth 1 submodules
    submodules_depth1 = re.findall(regex_depth1, individual)
    submodules_depth1.extend(re.findall(regex_execTree_depth1, individual))
    # find depth 2 submodules
    submodules_depth2 = re.findall(regex_depth2_exec, individual)
    submodules_depth2.extend(re.findall(regex_depth2, individual))
    submodules_depth2.extend(re.findall(regex_neg2, individual))

    ''' # another optimized way to do it. It is always O(n)
    submodules_depth1 = list(set(submodules_depth1)) # set conversion O(n)
    submodules_depth2 = list(set(submodules_depth2) - set(submodules_depth1))
    '''
    # optimized check # O(n)
    submodules_depth1_set = set(submodules_depth1) # "not in"/"in" is O(1) for a set, O(n) for a list
    submodules_depth2 = [module for module in submodules_depth2 if module not in submodules_depth1_set]
    
    '''
    # original check # p, b <= n, then it is O(n**3)
    for module in submodules_depth1: # O(p) with p = len(submodules_depth1)
        if module in submodules_depth2: #  worst case O(b) with b = len(submodules_depth2)
            submodules_depth2.remove(module) # worst case O(b) 
    '''
    return submodules_depth1, submodules_depth2

def extraction_list(individual): # O(n), everything is bounded by O(n) here
    ind = str(individual).replace(" ", "") # O(n)
    tree, _ = parse_expr(ind, 0) # O(n)
    submodules_depth1, submodules_depth2 = extract_nodes(tree, is_root=True) # O(n)
    
    d1_set = set(submodules_depth1) # O(n)
    submodules_depth2 = [mod for mod in submodules_depth2 if mod not in d1_set] # O(n)

    #print(f"ind: {ind}")
    #print(f"dp1: {submodules_depth1}")
    #print(f"dp2: {submodules_depth2}")

    return submodules_depth1, submodules_depth2

# extraction_list support
def parse_expr(ind, i): # O(n)
    j = i
    if ind[j] == '-':
        j += 1
    while j < len(ind) and (ind[j].isalnum() or ind[j] == '_'): # scan the ind O(n) shared
        j += 1
    token = ind[i:j]
    # shared mean they altogether have O(n) bc they iter the same object
    if j < len(ind) and ind[j] == "(":
        j += 1
        args = []
        while ind[j] != ")": # scan the ind O(n) shared
            arg, j = parse_expr(ind, j)
            args.append(arg)
            if ind[j] == ",":
                j += 1
        j += 1
        return (token, args), j
    return token, j

# extraction_list support
def is_operator(op): # O(1)
    return op in {"add", "sub", "neg", "mul", "div"} or (op.startswith("execTree") and op[8:].isdigit())

# extraction_list support
def is_flat(node): # O(k) with k the number of node args
    if not isinstance(node, tuple):
        return True
    _, args = node
    return all(not (isinstance(arg, tuple) and is_operator(arg[0])) for arg in args)

# extraction_list support
def node_to_str(node): # O(m) with m number of node characters
    if not isinstance(node, tuple):
        return node
    op, args = node
    return op + "(" + ",".join(node_to_str(arg) for arg in args) + ")"

# extraction_list support
def extract_nodes(node, is_root=False): # O(b) with b the number of nodes
    d1, d2 = [], []
    if isinstance(node, tuple):
        op, args = node
        if not is_root and is_operator(op):
            if is_flat(node):
                d1.append(node_to_str(node))
            else:
                d2.append(node_to_str(node))
        # scan all the nodes O(b)
        for arg in args:
            if isinstance(arg, tuple):
                a1, a2 = extract_nodes(arg, False)
                d1.extend(a1)
                d2.extend(a2)
    return d1, d2

# returns the population modules/subtrees
def get_modules(pop, extraction):
    my_dict1 = {}
    my_dict2 = {}
    for individual in pop:
        module_depth1, module_depth2 = extraction(individual) # extraction_tree or extraction_list
        fit = individual.fitness.values[0]
        for m in module_depth1:
            if m not in my_dict1:
                my_dict1[m] = [1, fit]
            else:
                my_dict1[m][0] += 1
                my_dict1[m][1] += fit

        for m in module_depth2:
            if m not in my_dict2:
                my_dict2[m] = [1, fit]
            else:
                my_dict2[m][0] += 1
                my_dict2[m][1] += fit

    return my_dict1, my_dict2

def view_hist(module_freq, depth):
    plt.subplots(figsize=(8, 6))
    keys = list(module_freq.keys())
    values = [module_freq[key][0] for key in keys]
    plt.bar(keys, values)
    plt.xlabel('Modules')
    plt.xticks(fontsize=8)
    plt.xticks(rotation=90)  
    plt.ylabel('Frequency')
    plt.title(f'Histogram of depth modules {depth}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.938, bottom=0.195)
    plt.show()

def view_hist_fitness_freq(modules_freq_fitness):
    fig, ax2 = plt.subplots()
    keys = list(modules_freq_fitness.keys())
    values_freq = []
    values_fitness = []
    for key in keys:
        values_freq.append(modules_freq_fitness[key][0])
        values_fitness.append(modules_freq_fitness[key][1])
    ax2.bar(keys, values_freq, color="blue", label="Frequency")
    ax1 = ax2.twinx()
    ax1.scatter(keys, values_fitness, color="red", label="Normalized fitness")
    ax1.set_xlabel("Modules")
    ax1.set_ylabel("Normalized fitness values")
    ax2.set_ylabel("Frequency values")
    fig.legend()
    plt.show()

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

    best_ind = None
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism
        if best_ind is not None:
            random_index = random.randint(0, len(population) - 1)
            if population[random_index].fitness < best_ind.fitness:
                population[random_index] = best_ind

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Elitism
        if halloffame is not None:
            best_ind = halloffame[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook.stream