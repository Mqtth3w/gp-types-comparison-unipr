# initially developed by Aeranna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti

import datetime
import math
import operator
import dill
import random
import re
import numpy as np
from deap import gp, creator, base, tools

from utils import *
from data_loader import load_dataset

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def modularGP(run, max_depth, generations, iterations, individual_to_keep, file_path, kernel_size, pop_size, const):
    KERNEL_SIZE = int(kernel_size)
    MAX_DEPTH = int(max_depth)
    MIN_DEPTH = 4
    N_GENERATIONS = int(generations)
    N_IND_TO_KEEP = int(individual_to_keep)
    N_ITERATIONS = int(iterations)
    N_POPULATION = int(pop_size)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(file_path)
    validation_f1, f1_score, statistic  = [], [], []

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
    
    def evalSet(individual, data, labels, type):
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, Y_labels_multi, y_predictions, rf = training_rf(new_train_set, train_labels, new_data, labels)
        print(f"Reached {mean_f1} F1 on {type} set")
        '''
        # confusion matrix
        conf_matrix = confusion_matrix(Y_labels_multi, y_predictions, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf.classes_)
        disp.plot()
        plt.show()
        '''
        return mean_f1

    def get_individuals_to_keep(pop, n, modules_depth1, modules_depth2):
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
           modules_freq[module][1] /= fitness_modules_sum if fitness_modules_sum != 0 else 1 # Cella put 0 here

        # graph view with frequencies and fitness associated with modules
        #view_hist_fitness_freq(modules_freq)

        # sorting by fitness 
        sorted_modules_fitness = dict(sorted(modules_freq.items(), key=lambda x: x[1][1], reverse=True))
        # selection of the n individuals to keep
        individuals_to_keep = [module for module in sorted_modules_fitness.keys() if len(module) > 1][:n]

        print("\nIndividuals to keep:")
        for i in range(len(individuals_to_keep)):
            print(f"{i}: {individuals_to_keep[i]}")

        return individuals_to_keep

    max_val=1.5e+100
    min_val=1.5e-100
    def mul(x, y):
        try:
            result = x * y
            if math.isinf(result):
                return max_val if result > 0 else min_val
            return result
        except Exception as e:
            print(f"Error in mul({x}, {y}): {e}")
            return max_val if (x > 0 and y > 0) or (x < 0 and y < 0) else min_val

    def protectedDiv(x, y):
        try:
            if y == 0:
                return 1
            result = x / y
            if math.isinf(result):
                return max_val if result > 0 else min_val
            return result
        except Exception as e:
            print(f"Error in protectedDiv({x}, {y}): {e}")
            return max_val if (x > 0 and y > 0) or (x < 0 and y < 0) else min_val
    
    pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant(f"rand101_{const}", lambda: random.randint(-1, 1))

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

    # we register new mutation and crossover operators
    toolbox.register("mate_subtree", gp.cxOnePoint)
    toolbox.register("mutate_subtree", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_point", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_insert", gp.mutInsert, pset=pset)

    # height limits for operators
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

    hof = [None] * N_ITERATIONS
    individuals_to_keep = [None] * N_ITERATIONS
    cntTree = 0

    new_pset_depth2 = gp.PrimitiveSet("MAIN", 4) 
    new_pset_depth2.addPrimitive(operator.add, 2)
    new_pset_depth2.addPrimitive(operator.sub, 2)
    new_pset_depth2.addPrimitive(mul, 2)
    new_pset_depth2.addPrimitive(protectedDiv, 2)
    new_pset_depth2.addPrimitive(operator.neg, 1)

    new_pset_depth1 = gp.PrimitiveSet("MAIN", 2) 
    new_pset_depth1.addPrimitive(operator.add, 2)
    new_pset_depth1.addPrimitive(operator.sub, 2)
    new_pset_depth1.addPrimitive(mul, 2)
    new_pset_depth1.addPrimitive(protectedDiv, 2)
    new_pset_depth1.addPrimitive(operator.neg, 1)
    
    # correct from here
    cnt_arg = 0
    cnt_iter = 0
    ind_best = None
    for cnt in range(N_ITERATIONS):
        pop = toolbox.population(n=N_POPULATION)
        
        # increase MAX_DEPTH only if fitness has deteriorated
        if cnt_iter > 0 and validation_f1[-1] < validation_f1[-2]:
            MAX_DEPTH += 2
        else:
            identity = gp.PrimitiveTree.from_string(f"ARG{int(KERNEL_SIZE/2)}", pset)
            identity = creator.Individual(identity)
            pop[random.randrange(N_POPULATION)] = identity
        
        # if I have a better individual from the previous generation, I reintroduce him.
        if ind_best is not None:
            pop[random.randrange(N_POPULATION)] = ind_best
        
        hof[cnt] = tools.HallOfFame(3)
        pop, log = eaSimple_elit(pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof[cnt], verbose=True)
        
        print(f"Best individual: {hof[cnt][0]}")
        
        # evaluation on training, validation and test sets
        f1_testSet = evalSet(hof[cnt][0], test_data, test_labels, "test")
        f1_valSet = evalSet(hof[cnt][0], val_data, val_labels, "validation")
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        statistic.append(log)
        
        modules_depth1, modules_depth2 = get_modules(pop)
        
        # selection of individuals to be maintained
        individuals_to_keep[cnt] = get_individuals_to_keep(pop, N_IND_TO_KEEP, modules_depth1, modules_depth2)
        
        # to replace ARG
        def replace(m):
            nonlocal cnt_arg
            output = f"ARG{cnt_arg}"
            cnt_arg = (cnt_arg + 1) % 3
            return output
        
        for i in range(len(individuals_to_keep[cnt])):
            cnt_arg = 0
            individuals_to_keep[cnt][i] = re.sub(r'ARG\d+', replace, individuals_to_keep[cnt][i])
        
        individuals_to_keep[cnt] = list(set(individuals_to_keep[cnt]))
        
        ind_best = hof[cnt][0]
        
        # transformation of strings into individuals
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = creator.Individual(
                gp.PrimitiveTree.from_string(individuals_to_keep[cnt][i], pset)
            )
        
        cnt1 = cntTree
        
        # adding modules as primitives
        for ind in individuals_to_keep[cnt]:
            depth_level = depth(str(ind))
            if depth_level == 2:
                func = gp.compile(expr=ind, pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cnt1}")
            elif depth_level == 1:
                func = gp.compile(expr=ind, pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cnt1}")
            else:
                print("MODULE ERROR: NOT OF DEPTH 1 OR 2")
            cnt1 += 1
        
        cnt_iter += 1
        
        for ind in individuals_to_keep[cnt]:
            depth_level = depth(str(ind))
            func = gp.compile(expr=ind, pset=(new_pset_depth2 if depth_level == 2 else new_pset_depth1))
            new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            cntTree += 1
    
    # save
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(f"best_individual_{current_time}.pickle", "wb") as f:
        dill.dump(hof[N_ITERATIONS-1][0], f)
    
    with open(f"pset_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)
    
    with open(f"parameters_{current_time}.txt", "w") as r:
        r.write(f"{const}\n{KERNEL_SIZE}\n")
    
    return validation_f1, f1_score, statistic