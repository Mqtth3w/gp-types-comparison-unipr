# Arianna Cella
import random
import re
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

########################################################################Arianna Cella
def extraction(individual):
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
    for module in submodules_depth1:
        if module in submodules_depth2:
            submodules_depth2.remove(module)
    return submodules_depth1, submodules_depth2

# returns the population modules/subtrees
def get_modules(pop):
    my_dict1 = {}
    my_dict2 = {}
    for individual in pop:
        module_depth1, module_depth2 = extraction(str(individual)) 

        for m in module_depth1:
            if m not in my_dict1:
                my_dict1[m] = [1, individual.fitness.values[0]]
            else:
                my_dict1[m][0] += 1
                my_dict1[m][1] += individual.fitness.values[0]

        for m in module_depth2:
            if m not in my_dict2:
                my_dict2[m] = [1, individual.fitness.values[0]]
            else:
                my_dict2[m][0] += 1
                my_dict2[m][1] += individual.fitness.values[0]

    return my_dict1, my_dict2 

def get_modules_individual(individual):
    modules = []
    module_depth1, module_depth2 = extraction(individual)
    modules.extend(module_depth1)
    modules.extend(module_depth2)
    return modules

def depth(string):
    string = str(string).replace(" ","")
    
    # regex for depth 1 submodules
    regex_depth1 = r'(?:add|sub|neg|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|-?\d+,-?\d+|[-A-Za-z0-9_]+,-?\d+|[-A-Za-z0-9_]+,[A-Za-z0-9_]+)\)'
    regex_execTree_depth1 = r'execTree\d+\((?:-?\d+|ARG\d+|[A-Za-z0-9_]+)(?:,-?\d+|,ARG\d+|,[A-Za-z0-9_]+){0,3}\)'

    # regex for depth 2 submodules
    regex_depth2 = r'(?:sub|add|mul|div|execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\)'
    regex_neg2 = r'(?:neg)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_depth1+r'|'+regex_execTree_depth1+r')\)'
    regex_depth2_exec = r'(?:execTree\d+)\((?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\,(?:-?\d+|[A-Za-z0-9_]+|\([^()]+\)|'+regex_execTree_depth1+r'|'+regex_depth1+r')\)'

    if (re.match(regex_depth1, string) or re.match(regex_execTree_depth1, string)):
        return 1
    if ((re.match(regex_depth2, string)) or (re.match(regex_neg2, string)) or (re.match(regex_depth2_exec, string))):
        return 2
    
    return None
########################################################################Arianna Cella


########################################################################Francesca Stefano
def extraction_stef(individual):
    individual = str(individual).replace(" ", "")
    
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
    
    return parse_expression(individual)

def get_modules_stef(pop):
    my_dict1 = {}
    my_dict2 = {}
    
    for p in pop:
        adj_list = extraction_stef(str(p))
        
        for node in adj_list:
            children = adj_list[node]
            if len(children) == 1:
                if node not in my_dict1:
                    my_dict1[node] = [1, p.fitness.values[0]]
                else:
                    my_dict1[node][0] += 1
                    my_dict1[node][1] += p.fitness.values[0]
            elif len(children) > 1:
                if node not in my_dict2:
                    my_dict2[node] = [1, p.fitness.values[0]]
                else:
                    my_dict2[node][0] += 1
                    my_dict2[node][1] += p.fitness.values[0]


    return my_dict1, my_dict2

def get_modules_individual_stef(individual):
    adj_list = extraction_stef(individual)
    module_depth1 = []
    module_depth2 = []
    
    for node in adj_list:
        children = adj_list[node]
        if len(children) == 1:
            module_depth1.append(node)
        elif len(children) > 1:
            module_depth2.append(node)
    
    return module_depth1, module_depth2

def depth_stef(individuo):
    adj_list = extraction(individuo)
    if not adj_list:
        print("Error: adj_list is empty.")
        return 0
    
    def max_depth(node):
        if node not in adj_list or not adj_list[node]:
            return 1
        else:
            return 1 + max(max_depth(child) for child in adj_list[node])
    
    return max(max_depth(node) for node in adj_list)
########################################################################Francesca Stefano

# Cella
# displays forms that occur more frequently than 5
def view_hist(module_freq, depth):
    plt.subplots(figsize=(8, 6))
    keys = list(module_freq.keys())
    values = [module_freq[key][0] for key in keys]
    plt.bar(keys,values)
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

def eaSimple_elit(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    """An improved version of eaSimple with proper elitism handling"""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # evaluate initial population
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

    for gen in range(1, ngen + 1):
        # select the next generation
        offspring = toolbox.select(population, len(population))
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # update hall of fame and keep the best individual
        if halloffame is not None:
            halloffame.update(offspring)
            best_ind = halloffame[0]  # Best individual from HoF

            # ensure the best individual is in the new generation
            if best_ind not in offspring:
                worst_idx = min(range(len(offspring)), key=lambda i: offspring[i].fitness.values)
                offspring[worst_idx] = toolbox.clone(best_ind)

        # replace old population with new offspring
        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook