# initially developed by Arianna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti

import datetime
import math
import operator
import dill
from deap import gp, creator, base
from utils import *
from data_loader import load_dataset
import matplotlib
matplotlib.use('TkAgg')
import pathos.multiprocessing as multiprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def modularGP_CellaMethod(file_path, outbox, MAX_DEPTH, N_GENERATIONS, N_POPULATION, N_ITERATIONS, N_IND_TO_KEEP, KERNEL_SIZE, const):
    MIN_DEPTH = 4

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
    
    def evalSet(individual, data, labels, type): # test/validation
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, Y_labels_multi, y_predictions, rf = training_rf(new_train_set, train_labels, new_data, labels)
        outbox.insert(tk.END, f"Reached {mean_f1} F1 on {type} set\n")
        #print(f"Reached {mean_f1} F1 on {type} set") 
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

        outbox.insert(tk.END, "\nIndividuals to keep:\n")
        #print("\nIndividuals to keep:")
        for i in range(len(individuals_to_keep)):
            #print(f"{i}: {individuals_to_keep[i]}")
            outbox.insert(tk.END, f"{i}: {individuals_to_keep[i]}\n")

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
    new_pset_depth2.addPrimitive(mul, 2)
    new_pset_depth2.addPrimitive(protectedDiv, 2)
    new_pset_depth2.addPrimitive(operator.neg, 1)

    new_pset_depth1 = gp.PrimitiveSet("MAIN", 2) 
    new_pset_depth1.addPrimitive(operator.add, 2)
    new_pset_depth1.addPrimitive(operator.sub, 2)
    new_pset_depth1.addPrimitive(mul, 2)
    new_pset_depth1.addPrimitive(protectedDiv, 2)
    new_pset_depth1.addPrimitive(operator.neg, 1)
    
    # create a function to replace the ARG number with the specific value (cnt_arg 0 -> 1 -> 2 -> 0)
    def replace(_):
        nonlocal cnt_arg
        output = f"ARG{cnt_arg}"
        cnt_arg = (cnt_arg + 1) % 3
        print(outbox)
        return output
    
    hof = [None] * N_ITERATIONS
    individuals_to_keep = [None] * N_ITERATIONS
    cntTree = 0
    cnt_arg = 0
    ind_best = None

    for cnt in range(N_ITERATIONS):
        pop = toolbox.population(n=N_POPULATION)
        
        if cnt == 0:
            identity = gp.PrimitiveTree.from_string(f"ARG{KERNEL_SIZE // 2}", pset)
            identity = creator.Individual(identity)
            pop[random.randrange(N_POPULATION // 2)] = identity
        
        if ind_best is not None:
            pop[random.randrange(N_POPULATION // 2, N_POPULATION)] = ind_best

        hof[cnt] = tools.HallOfFame(3)
        pop, log = eaSimple_elit(outbox, pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof[cnt], verbose=True)

        outbox.insert(tk.END, f"Best individual: {hof[cnt][0]}\n")
        #print(f"Best individual: {hof[cnt][0]}")
        
        # evaluation on training, validation and test sets
        f1_testSet = evalSet(hof[cnt][0], test_data, test_labels, "test")
        f1_valSet = evalSet(hof[cnt][0], val_data, val_labels, "validation")
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        statistic.append(log)
        
        modules_depth1, modules_depth2 = get_modules_tree(pop)
        
        # frequency charts
        #view_hist(modules_depth1, 1)
        #view_hist(modules_depth2, 2)

        # selection of individuals to be maintained
        individuals_to_keep[cnt] = get_individuals_to_keep(N_IND_TO_KEEP, modules_depth1, modules_depth2)
        
        cnt_arg = 0
        for i in range(len(individuals_to_keep[cnt])):
            #cnt_arg = 0 # nosense it will be always 0
            individuals_to_keep[cnt][i] = re.sub(r'ARG\d+', replace, individuals_to_keep[cnt][i])

        individuals_to_keep[cnt] = list(set(individuals_to_keep[cnt]))
        
        ind_best = hof[cnt][0]

        # transformation of strings into individuals
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = creator.Individual(
                gp.PrimitiveTree.from_string(individuals_to_keep[cnt][i], pset)
            )
        
        cnt1 = cntTree
        
        # adds to the primitives the modules to be maintained in the next interaction
        for ind in individuals_to_keep[cnt]:
            depth_level = depth_tree(str(ind))
            if depth_level == 2:
                func = gp.compile(expr=ind, pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cnt1}")
            elif depth_level == 1:
                func = gp.compile(expr=ind, pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cnt1}")
            else:
                outbox.insert(tk.END, "MODULE ERROR: NOT OF DEPTH 1 OR 2\n")
                #print("MODULE ERROR: NOT OF DEPTH 1 OR 2")
            cnt1 += 1

        for ind in individuals_to_keep[cnt]:
            depth_level = depth_tree(str(ind))
            func = gp.compile(expr=ind, pset=(new_pset_depth2 if depth_level == 2 else new_pset_depth1))
            new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            cntTree += 1

    # save
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"modularGP_CellaMethod_best_individual_{current_time}.pickle", "wb") as f:
        dill.dump(hof[N_ITERATIONS-1][0], f)
    with open(f"modularGP_CellaMethod_pset_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)
    
    return hof[N_ITERATIONS-1][0], validation_f1, f1_score, statistic

def modularGP_StefanoMethod(file_path, outbox, MAX_DEPTH, N_GENERATIONS, N_POPULATION, N_ITERATIONS, N_IND_TO_KEEP, KERNEL_SIZE, const):
    MIN_DEPTH = 4

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
        num_nodes = individual.height
        K = 0.01
        fitness = f1_validation / (1 + K * num_nodes)
        
        return (fitness,)
    
    def evalSet(individual, data, labels, type): # test/validation
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set = convolution(func, train_data, KERNEL_SIZE)
        new_data = convolution(func, data, KERNEL_SIZE)
        mean_f1, Y_labels_multi, y_predictions, rf = training_rf(new_train_set, train_labels, new_data, labels)
        outbox.insert(tk.END, f"Reached {mean_f1} F1 on {type} set\n")
        #print(f"Reached {mean_f1} F1 on {type} set") 
        '''
        # confusion matrix
        conf_matrix = confusion_matrix(Y_labels_multi, y_predictions, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf.classes_)
        disp.plot()
        plt.show()
        '''
        return mean_f1

    def get_individuals_to_keep(n, modules_depth1, modules_depth2):
        '''
        outbox.insert(tk.END, "Modules depth 1:\n")
        #print("Modules depth 1:")
        for key, value in modules_depth1.items():
            #print(f"{key}: {value}")
            outbox.insert(tk.END, f"{key}: {value}\n")

        outbox.insert(tk.END, "Modules depth 2:\n")
        #print("Modules depth 2:")
        for key, value in modules_depth2.items():
            #print(f"{key}: {value}")
            outbox.insert(tk.END, f"{key}: {value}\n")
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
        individuals_to_keep = [module for module in sorted_modules_fitness.keys() if len(module) > 1][:n]

        outbox.insert(tk.END, "\nIndividuals to keep:\n")
        #print("\nIndividuals to keep:")
        for i, module in enumerate(individuals_to_keep):
            #print(f"{i}: {module}")
            outbox.insert(tk.END, f"{i}: {module}\n")

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
            outbox.insert(tk.END, f"Error in mul({x}, {y}): {e}\n")
            #print(f"Error in mul({x}, {y}): {e}")
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
            outbox.insert(tk.END, f"Error in protectedDiv({x}, {y}): {e}\n")
            #print(f"Error in protectedDiv({x}, {y}): {e}")
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
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # register new op
    toolbox.register("mutate_insert", gp.mutInsert, pset=pset)
    # duplicated 
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
    new_pset_depth2.addPrimitive(mul, 2)
    new_pset_depth2.addPrimitive(protectedDiv, 2)
    new_pset_depth2.addPrimitive(operator.neg, 1)

    new_pset_depth1 = gp.PrimitiveSet("MAIN", 2)
    new_pset_depth1.addPrimitive(operator.add, 2)
    new_pset_depth1.addPrimitive(operator.sub, 2)
    new_pset_depth1.addPrimitive(mul, 2)
    new_pset_depth1.addPrimitive(protectedDiv, 2)
    new_pset_depth1.addPrimitive(operator.neg, 1)

    hof = [None] * N_ITERATIONS
    individuals_to_keep = [None] * N_ITERATIONS
    cntTree = 0
    ind_best = None

    def adjust_probabilities(population, gen):
        """Adjust probabilities of mutation and crossover based on population diversity or fitness stagnation."""
        diversity_threshold = 0.2
        fitness_improvement_threshold = 0.01
        
        fitness_values = [ind.fitness.values[0] for ind in population]
        avg_fitness = np.mean(fitness_values)
        #max_fitness = np.max(fitness_values) # it was unused
        
        if gen > 1:
            prev_avg_fitness = np.mean([ind.fitness.values[0] for ind in prev_population])
            fitness_improvement = avg_fitness - prev_avg_fitness
        else:
            fitness_improvement = fitness_improvement_threshold + 1
        
        diversity = len(set(fitness_values)) / len(fitness_values)
        
        if diversity < diversity_threshold or fitness_improvement < fitness_improvement_threshold:
            cxpb = 0.3 # lower crossover rate
            mutpb = 0.3 # higher mutation rate
        else:
            cxpb = 0.6 # higher crossover rate
            mutpb = 0.1 # lower mutation rate
        
        prev_population[:] = population
        return cxpb, mutpb
        
    prev_population = []

    cxpb, mutpb = 0.5, 0.1
    for cnt in range(N_ITERATIONS):
        pop = toolbox.population(n=N_POPULATION)

        if cnt == 0:
            identity = gp.PrimitiveTree.from_string(f"ARG{KERNEL_SIZE // 2}", pset)
            identity = creator.Individual(identity)
            pop[random.randrange(N_POPULATION // 2)] = identity
        
        if ind_best is not None:
            pop[random.randrange(N_POPULATION // 2, N_POPULATION)] = ind_best

        hof[cnt] = tools.HallOfFame(3)
        
        for gen in range(N_GENERATIONS):
            pop = varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof[cnt].update(pop)
            cxpb, mutpb = adjust_probabilities(pop, gen)
            pop = toolbox.select(pop, len(pop))
            log = mstats.compile(pop)
            statistic.append(log)

        outbox.insert(tk.END, f"Best individual: {hof[cnt][0]}\n")
        #print(f"Best individual: {hof[cnt][0]}")
        f1_testSet = evalSet(hof[cnt][0], test_data, test_labels, "test")
        f1_valSet = evalSet(hof[cnt][0], val_data, val_labels, "validation")
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        
        modules_depth1, modules_depth2 = get_modules_list(pop)
        #view_hist(modules_depth1, 1)
        #view_hist(modules_depth2, 2)
        #view_hist_fitness_freq(modules_depth1)
        #view_hist_fitness_freq(modules_depth2)
        
        individuals_to_keep[cnt] = get_individuals_to_keep(N_IND_TO_KEEP, modules_depth1, modules_depth2)
    
        cnt1 = cntTree
    
        # adds to the primitives the modules to be maintained in the next interaction
        for ind in individuals_to_keep[cnt]:
            depth_level = depth_tree(str(ind))
            if depth_level == 2:
                func = gp.compile(expr=ind, pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cnt1}")
            elif depth_level == 1:
                func = gp.compile(expr=ind, pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cnt1}")
            else:
                outbox.insert(tk.END, "MODULE ERROR: NOT OF DEPTH 1 OR 2\n")
                #print("MODULE ERROR: NOT OF DEPTH 1 OR 2")
            cnt1 += 1

        for ind in individuals_to_keep[cnt]:
            depth_level = depth_tree(str(ind))
            func = gp.compile(expr=ind, pset=(new_pset_depth2 if depth_level == 2 else new_pset_depth1))
            new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            cntTree += 1

    # save
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"modularGP_StefanoMethod_best_individual_{current_time}.pickle", "wb") as f:
        dill.dump(hof[N_ITERATIONS-1][0], f)
    with open(f"modularGP_StefanoMethod_pset_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)

    return hof[N_ITERATIONS-1][0], validation_f1, f1_score, statistic

def classicalGP(file_path, outbox, MAX_DEPTH, N_GENERATIONS, N_POPULATION, N_ITERATIONS, N_IND_TO_KEEP, KERNEL_SIZE, const):
    pass

def classicalGP(file_path, outbox, MAX_DEPTH, N_GENERATIONS, N_POPULATION, _, __, KERNEL_SIZE, const):
    MIN_DEPTH = 4

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(file_path)
    validation_f1, f1_score, statistic  = [], [], []

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
        outbox.insert(tk.END, f"Reached {mean_f1} F1 on {type} set\n")
        return mean_f1
    
    pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(lambda x, y: x / y if y != 0 else 1, 2)
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
    pop = toolbox.population(n=N_POPULATION)
    pop, log = eaSimple_elit(outbox, pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof, verbose=True)
    
    outbox.insert(tk.END, f"Best individual: {hof[0]}\n")
    f1_testSet = evalSet(hof[0], test_data, test_labels, "test")
    f1_valSet = evalSet(hof[0], val_data, val_labels, "validation")
    validation_f1.append(f1_valSet)
    f1_score.append(f1_testSet)
    statistic.append(log)
    
    return hof[0], validation_f1, f1_score, statistic




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
# anyway in the enached GP method they applied the purpose is limit the depth and work on the solutions
# in fact they also used staticLimit so the editing of MAX_DEPTH was a random operation
'''