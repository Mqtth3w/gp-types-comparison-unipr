# initially developed by Arianna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
import functools
import operator
import dill
from deap import gp, creator, base
from utils import *
from data_loader import load_dataset
import os
#import pathos.multiprocessing as multiprocessing
#from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def modularGP_CellaMethod(current_time, file_path, verbose, MAX_DEPTH, N_GENERATIONS, N_POPULATION, N_ITERATIONS, N_IND_TO_KEEP, KERNEL_SIZE, const):
    MIN_DEPTH = 4

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(file_path)
    validation_f1, f1_score, statistics  = [], [], []

    def evalTrainingSet(individual):
        # compilation of the individual
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_val_set = convolution(func, val_data, KERNEL_SIZE)

        # model training and evaluation
        f1_validation, _, _, _ = training_rf(new_train_set, train_labels, new_val_set, val_labels)

        # fitness calculation with penalty proportional to the number of nodes in the tree
        num_nodes = len(individual)
        K = 1e-4
        fitness = f1_validation / (1 + K * num_nodes)
        
        return (fitness,)
    
    def evalSet(individual, data, labels, type): # test/validation
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, Y_labels_multi, y_predictions, rf = training_rf(new_train_set, train_labels, new_data, labels)
        print(f"(modularGP_CellaMethod) Reached {mean_f1} F1 on {type} set") 
        '''
        # confusion matrix
        conf_matrix = confusion_matrix(Y_labels_multi, y_predictions, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf.classes_)
        disp.plot()
        plt.show()
        '''
        return mean_f1
    
    def get_individuals_to_keep(n, modules_depth1, modules_depth2):
        # sorting by frequency
        sorted_modules_1 = dict(sorted(modules_depth1.items(), key=lambda x: x[1][0], reverse=True))
        sorted_modules_2 = dict(sorted(modules_depth2.items(), key=lambda x: x[1][0], reverse=True))
        # I extract the first 5 elements (the ones that appear the most), in this way I will be able to maintain a maximum of 10 individuals
        modules_freq = {}
        for key, value in list(sorted_modules_1.items())[:5]:
            value[1] /= value[0] # fitness mean
            modules_freq[key] = value
        for key, value in list(sorted_modules_2.items())[:5]:
            value[1] /= value[0]
            modules_freq[key] = value
        
        # fitness normalization
        fitness_modules_sum = sum(v[1] for v in modules_freq.values())
        for module in modules_freq:
            if fitness_modules_sum != 0:
                modules_freq[module][1] /= fitness_modules_sum
            else:
                modules_freq[module][1] = 0

        # graph view with frequencies and fitness associated with modules
        #view_hist_fitness_freq(modules_freq)

        # sorting by fitness 
        sorted_modules_fitness = dict(sorted(modules_freq.items(), key=lambda x: x[1][1], reverse=True))

        # selection of the n individuals to keep
        individuals_to_keep = []
        for module in sorted_modules_fitness.keys():
            if len(individuals_to_keep) < n:
                individuals_to_keep.append(module)
            else:
                break

        #print("\n(modularGP_CellaMethod) Individuals to keep:")
        #for i in range(len(individuals_to_keep)):
            #print(f"(modularGP_CellaMethod) {i}: {individuals_to_keep[i]}")

        return individuals_to_keep
    
    pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant(f"rand101_{const}", functools.partial(random.randint, -1, 1))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalTrainingSet)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

    #toolbox.register("map", multiprocessing.Pool().map)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    new_pset_depth2 = gp.PrimitiveSet("MAIN", 4) 
    new_pset_depth2.addPrimitive(operator.add, 2)
    new_pset_depth2.addPrimitive(operator.sub, 2)
    new_pset_depth2.addPrimitive(operator.mul, 2)
    new_pset_depth2.addPrimitive(div, 2)
    new_pset_depth2.addPrimitive(operator.neg, 1)

    new_pset_depth1 = gp.PrimitiveSet("MAIN", 2) 
    new_pset_depth1.addPrimitive(operator.add, 2)
    new_pset_depth1.addPrimitive(operator.sub, 2)
    new_pset_depth1.addPrimitive(operator.mul, 2)
    new_pset_depth1.addPrimitive(div, 2)
    new_pset_depth1.addPrimitive(operator.neg, 1)
    
    def count_nodes(individual):
        tot_nodes = 0
        for node in individual:
            if isinstance(node, gp.Primitive) and node.name.startswith("execTree"):
                original_tree = pset.mapping[node.name]
                tot_nodes += len(original_tree)
            else:
                tot_nodes += 1
        return tot_nodes
    
    # (cnt_arg 0 -> 1 -> 0)
    def replace(_):
        nonlocal cnt_arg
        output = f"ARG{cnt_arg}"
        cnt_arg = (cnt_arg + 1) % 2
        return output
    
    hof = [None] * N_ITERATIONS
    individuals_to_keep = [None] * N_ITERATIONS
    cntTree = 0
    best_ind = None
    
    for cnt in range(N_ITERATIONS):
        print(f"(modularGP_CellaMethod) iter {cnt} strated...")
        pop = toolbox.population(n=N_POPULATION)
        
        if cnt == 0:
            identity = gp.PrimitiveTree.from_string(f"ARG{KERNEL_SIZE // 2}", pset)
            identity = creator.Individual(identity)
            pop[random.randrange(N_POPULATION // 2)] = identity
        
        if best_ind is not None:
            pop[random.randrange(N_POPULATION // 2, N_POPULATION)] = best_ind
        
        hof[cnt] = tools.HallOfFame(3)
        pop, log = eaSimple_elit(pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof[cnt], verbose=verbose)
        
        best_ind = hof[cnt][0]
        print(f"(modularGP_CellaMethod, iter:{cnt}) Best individual: {best_ind}\n")
        
        # evaluation on training, validation and test sets
        f1_testSet = evalSet(best_ind, test_data, test_labels, "test")
        f1_valSet = evalSet(best_ind, val_data, val_labels, "validation")
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        statistics.append(log)
        
        modules_depth1, modules_depth2 = get_modules(pop, extraction_tree)
        
        # frequency charts
        #view_hist(modules_depth1, 1)
        #view_hist(modules_depth2, 2)
        
        # selection of individuals to be maintained
        individuals_to_keep[cnt] = get_individuals_to_keep(N_IND_TO_KEEP, modules_depth1, modules_depth2)
        
        cnt_arg = 0
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = re.sub(r'ARG\d+', replace, individuals_to_keep[cnt][i])
        
        #individuals_to_keep[cnt] = list(set(individuals_to_keep[cnt]))
        
        # transformation of strings into individuals
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = creator.Individual(
                gp.PrimitiveTree.from_string(individuals_to_keep[cnt][i], pset)
            )
        
        # adds to the primitives the modules to be maintained in the next interation
        for ind in individuals_to_keep[cnt]:
            if ind.height == 2:
                func = gp.compile(expr=ind, pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cntTree}")
                new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            elif ind.height == 1:
                func = gp.compile(expr=ind, pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cntTree}")
                new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            else: # it means the module is a terminal
                print("(modularGP_CellaMethod) MODULE ERROR: NOT OF DEPTH 1 OR 2") # should never happen
            cntTree += 1
    
    # save
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{script_dir}/modularGP_CellaMethod_best_individual_run{const}_{current_time}.pickle", "wb") as f:
        dill.dump(best_ind, f)
    with open(f"{script_dir}/modularGP_CellaMethod_pset_run{const}_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)
    best_ind_len = count_nodes(best_ind)
    return best_ind, best_ind_len, validation_f1, f1_score, statistics

def modularGP_StefanoMethod(current_time, file_path, verbose, MAX_DEPTH, N_GENERATIONS, N_POPULATION, N_ITERATIONS, N_IND_TO_KEEP, KERNEL_SIZE, const):
    MIN_DEPTH = 4

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(file_path)
    validation_f1, f1_score, statistics  = [], [], []

    def evalTrainingSet(individual, pset, train_data, val_data):
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_val_set = convolution(func, val_data, KERNEL_SIZE)
        f1_validation, _, _, _ = training_rf(new_train_set, train_labels, new_val_set, val_labels)
        num_nodes = len(individual) #it was individual.height that is the depth
        K = 1e-4
        fitness = f1_validation / (1 + K * num_nodes)
        return (fitness,)
    
    def evalSet(individual, data, labels, type): # test/validation
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, _, _, _ = training_rf(new_train_set, train_labels, new_data, labels)
        print(f"(modularGP_StefanoMethod) Reached {mean_f1} F1 on {type} set") 
        return mean_f1

    def get_individuals_to_keep(n, modules_depth1, modules_depth2):
        '''
        print("(modularGP_StefanoMethod) Modules depth 1:")
        for key, value in modules_depth1.items():
            print(f"(modularGP_StefanoMethod) {key}: {value}")

        print("(modularGP_StefanoMethod) Modules depth 2:")
        for key, value in modules_depth2.items():
            print(f"(modularGP_StefanoMethod) {key}: {value}")
        '''
        sorted_modules_1 = dict(sorted(modules_depth1.items(), key=lambda x: x[1][0], reverse=True))
        sorted_modules_2 = dict(sorted(modules_depth2.items(), key=lambda x: x[1][0], reverse=True))

        modules_freq = {}
        for key, value in list(sorted_modules_1.items())[:5]:
            value[1] /= value[0]
            modules_freq[key] = value

        for key, value in list(sorted_modules_2.items())[:5]:
            value[1] /= value[0]
            modules_freq[key] = value

        # fitness normalization
        fitness_modules_sum = sum(v[1] for v in modules_freq.values())
        for module in modules_freq:
            if fitness_modules_sum != 0:
                modules_freq[module][1] /= fitness_modules_sum
            else:
                modules_freq[module][1] = 0

        sorted_modules_fitness = dict(sorted(modules_freq.items(), key=lambda x: x[1][1], reverse=True))
        
        # selection of the n individuals to keep
        individuals_to_keep = []
        for module in sorted_modules_fitness.keys():
            if len(individuals_to_keep) < n:
                individuals_to_keep.append(module)
            else:
                break
        '''
        print("\n(modularGP_StefanoMethod) Individuals to keep:")
        for i, module in enumerate(individuals_to_keep):
            print(f"(modularGP_StefanoMethod) {i}: {module}")
        '''
        return individuals_to_keep

    pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant(f"rand101_{const}", functools.partial(random.randint, -1, 1))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalTrainingSet)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # register new "op"
    toolbox.register("mutate_insert", gp.mutInsert, pset=pset)
    # duplicated? It may be useful only for calculating the probabilities dinamically
    toolbox.register("mate_subtree", gp.cxOnePoint)
    toolbox.register("mutate_subtree", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_point", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # op height limits
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    new_pset_depth2 = gp.PrimitiveSet("MAIN", 4)
    new_pset_depth2.addPrimitive(operator.add, 2)
    new_pset_depth2.addPrimitive(operator.sub, 2)
    new_pset_depth2.addPrimitive(operator.mul, 2)
    new_pset_depth2.addPrimitive(div, 2)
    new_pset_depth2.addPrimitive(operator.neg, 1)

    new_pset_depth1 = gp.PrimitiveSet("MAIN", 2)
    new_pset_depth1.addPrimitive(operator.add, 2)
    new_pset_depth1.addPrimitive(operator.sub, 2)
    new_pset_depth1.addPrimitive(operator.mul, 2)
    new_pset_depth1.addPrimitive(div, 2)
    new_pset_depth1.addPrimitive(operator.neg, 1)

    def count_nodes(individual):
        tot_nodes = 0
        for node in individual:
            if isinstance(node, gp.Primitive) and node.name.startswith("execTree"):
                original_tree = pset.mapping[node.name]
                tot_nodes += len(original_tree)
            else:
                tot_nodes += 1
        return tot_nodes
    
    def replace(_):
        nonlocal cnt_arg
        output = f"ARG{cnt_arg}"
        cnt_arg = (cnt_arg + 1) % 2
        return output
    
    hof = [None] * N_ITERATIONS
    individuals_to_keep = [None] * N_ITERATIONS
    cntTree = 0
    best_ind = None

    def adjust_probabilities(population):
        """Adjust probabilities of mutation and crossover based on population diversity or fitness stagnation."""
        nonlocal cxpb, mutpb, prev_population
        diversity_threshold = 0.2
        fitness_improvement_threshold = 0.01
        
        fitness_values = [ind.fitness.values[0] for ind in population]
        avg_fitness = np.mean(fitness_values)
        
        if prev_population:
            prev_avg_fitness = np.mean([ind.fitness.values[0] for ind in prev_population])
            fitness_improvement = avg_fitness - prev_avg_fitness
        else:
            fitness_improvement = fitness_improvement_threshold + 1
        
        diversity = len(set(fitness_values)) / len(fitness_values)
        
        # adjust probabilities with clamping to valid ranges [0, 1]
        if diversity < diversity_threshold or fitness_improvement < fitness_improvement_threshold:
            cxpb = max(cxpb - 0.05, 0.0) # decrease crossover
            mutpb = min(mutpb + 0.05, 1.0) # increase mutation
        else:
            cxpb = min(cxpb + 0.05, 1.0) # revert crossover
            mutpb = max(mutpb - 0.05, 0.0) # revert mutation
        
        prev_population[:] = population
        
    prev_population = []

    cxpb, mutpb = 0.5, 0.1
    for cnt in range(N_ITERATIONS):
        print(f"(modularGP_StefanoMethod) iter {cnt} strated...")
        pop = toolbox.population(n=N_POPULATION)

        if cnt == 0:
            identity = gp.PrimitiveTree.from_string(f"ARG{KERNEL_SIZE // 2}", pset)
            identity = creator.Individual(identity)
            pop[random.randrange(N_POPULATION // 2)] = identity
        
        if best_ind is not None:
            pop[random.randrange(N_POPULATION // 2, N_POPULATION)] = best_ind

        hof[cnt] = tools.HallOfFame(3)
        pop, log = eaSimple_elit(pop, toolbox, cxpb, mutpb, N_GENERATIONS, stats=mstats, halloffame=hof[cnt], verbose=verbose)

        adjust_probabilities(pop)

        best_ind = hof[cnt][0]
        print(f"(modularGP_StefanoMethod, iter:{cnt}) Best individual: {best_ind}\n")
        f1_testSet = evalSet(best_ind, test_data, test_labels, "test")
        f1_valSet = evalSet(best_ind, val_data, val_labels, "validation")
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        statistics.append(log)
        
        modules_depth1, modules_depth2 = get_modules(pop, extraction_list)
        #view_hist(modules_depth1, 1)
        #view_hist(modules_depth2, 2)
        #view_hist_fitness_freq(modules_depth1)
        #view_hist_fitness_freq(modules_depth2)
        
        individuals_to_keep[cnt] = get_individuals_to_keep(N_IND_TO_KEEP, modules_depth1, modules_depth2)

        cnt_arg = 0
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = re.sub(r'ARG\d+', replace, individuals_to_keep[cnt][i])
        
        individuals_to_keep[cnt] = list(set(individuals_to_keep[cnt]))
        
        # transformation of strings into individuals
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = creator.Individual(
                gp.PrimitiveTree.from_string(individuals_to_keep[cnt][i], pset)
            )
        
        # adds to the primitives the modules to be maintained in the next interation
        for ind in individuals_to_keep[cnt]:
            #print(f"indkeep: {ind}")
            if ind.height == 2:
                func = gp.compile(expr=ind, pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cntTree}")
                new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            elif ind.height == 1:
                func = gp.compile(expr=ind, pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cntTree}")
                new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            else: # it means the module is a terminal
                print("(modularGP_StefanoMethod) MODULE ERROR: NOT OF DEPTH 1 OR 2") # should never happen
            cntTree += 1

    # save
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{script_dir}/modularGP_StefanoMethod_best_individual_run{const}_{current_time}.pickle", "wb") as f:
        dill.dump(best_ind, f)
    with open(f"{script_dir}/modularGP_StefanoMethod_pset_run{const}_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)
    best_ind_len = count_nodes(best_ind)
    return best_ind, best_ind_len, validation_f1, f1_score, statistics

def classicalGP(current_time, file_path, verbose, MAX_DEPTH, N_GENERATIONS, N_POPULATION, _, __, KERNEL_SIZE, const):
    MIN_DEPTH = 4

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(file_path)
    validation_f1, f1_score, statistics  = [], [], []

    def evalTrainingSet(individual):
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_val_set = convolution(func, val_data, KERNEL_SIZE)
        f1_validation, _, _, _ = training_rf(new_train_set, train_labels, new_val_set, val_labels)
        num_nodes = len(individual)
        K = 1e-4
        fitness = f1_validation / (1 + K * num_nodes)
        return (fitness,)

    def evalSet(individual, data, labels, type):
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, _, _, _ = training_rf(new_train_set, train_labels, new_data, labels)
        print(f"(classicalGP) Reached {mean_f1} F1 on {type} set") 
        return mean_f1

    pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant(f"rand101_{const}", functools.partial(random.randint, -1, 1))
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    toolbox.register("evaluate", evalTrainingSet)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    print(f"(classicalGP) evolution strated...")
    pop = toolbox.population(n=N_POPULATION)
    pop, log = eaSimple_elit(pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof, verbose=verbose)
    
    print(f"(classicalGP) Best individual: {hof[0]}\n")
    f1_testSet = evalSet(hof[0], test_data, test_labels, "test")
    f1_valSet = evalSet(hof[0], val_data, val_labels, "validation")
    validation_f1.append(f1_valSet)
    f1_score.append(f1_testSet)
    statistics.append(log)
    
    # save
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{script_dir}/classicalGP_best_individual_{current_time}.pickle", "wb") as f:
        dill.dump(hof[0], f)
    with open(f"{script_dir}/classicalGP_pset_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)

    return hof[0], len(hof[0]), validation_f1, f1_score, statistics


''' # check the doc for more details
# here a demostration that editing c (MAX_DEPTH) does not affect the successive calls 
# it wiil be seen always as 1 (the value passed at the moment of registration)
# in the "Reference Code" by Cella/Stefano c is MAX_DEPTH
def func(a, b, c=3):
    print(a, b, c)

c = 1 
tools = base.Toolbox()
tools.register("myFunc", func, 2, b=3, c=c)
tools.myFunc()
c = 2
tools.myFunc()

# final output:
# 2 3 1
# 2 3 1 # so it is not 2 3 2
# to solve it you need to do tools.myFunc(c=c)
# they did not it so they always run the program with the fixed MAX_DEPTH initial value
# anyway in the enached GP method the applied the purpose is to limit the depth and work on the solutions
# in fact they also used staticLimit so the editing of MAX_DEPTH was a random operation
'''