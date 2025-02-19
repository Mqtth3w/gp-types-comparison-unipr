import collections
import datetime
import math
import operator
import pickle
import dill
import random
import re
import numpy as np
from deap import gp, creator, base, tools, algorithms
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pathos.multiprocessing as multiprocessing

from utils_functions import convolution, depth, eaSimple_elit, get_modules_individual, training_RF, get_modules, view_hist1,view_hist2, view_hist_fitness_freq
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils_import_data import get_dataset
cnt_arg=0 
cnt_iter=0

def modularGP(run,max_depth,generations,iterations,individual_to_keep,file_path,kernel_size,pop_size,const):
    KERNEL_SIZE = int(kernel_size)
    MAX_DEPTH = int(max_depth)
    MIN_DEPTH = 4
    N_GENERATIONS = int(generations)
    N_IND_TO_KEEP = int(individual_to_keep)
    N_ITERATIONS=int(iterations)
    N_POPOLATION=int(pop_size)

    train_data, train_labels, test_data, test_labels, data_val, labels_val = get_dataset(file_path)
    validation_f1=[]
    f1_score=[]
    statistic=[]


    def evalTrainingSet(individual):
        global cnt_iter
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        num_nodes=len(individual)
        new_train_set=convolution(func,train_data,KERNEL_SIZE,cnt_iter)
        new_val_set=convolution(func,data_val,KERNEL_SIZE,cnt_iter)
        f1_validation,l,p,rf=training_RF(new_train_set,train_labels, new_val_set, labels_val)
        # Calcolo della fitness con penalizzazione proporzionale al numero di nodi dell'albero
        K = 0.00001 # Esempio di valore per il parametro K
        fitness = f1_validation / (1 + K * num_nodes)
        return (fitness,)


    def evalTestSet(individual):
        global cnt_iter
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set=convolution(func,train_data, KERNEL_SIZE,cnt_iter)
        new_test_set=convolution(func,test_data,KERNEL_SIZE,cnt_iter)
        f1_t, Y_labels_multi, y_predictions, rf=training_RF(new_train_set,train_labels, new_test_set, test_labels)
        print(f"Reached {f1_t} F1 on test set ")
        # Calcolare la matrice di confusione
      #  conf_matrix = confusion_matrix(Y_labels_multi, y_predictions, labels=rf.classes_)
       # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf.classes_)
       # disp.plot()
       # plt.show()
        return f1_t
    
    def evalValidationSet(individual):
        global cnt_iter
        clf = gp.PrimitiveTree(individual)
        func = gp.compile(clf, pset)
        new_train_set=convolution(func,train_data,KERNEL_SIZE,cnt_iter)
        new_val_set=convolution(func,data_val,KERNEL_SIZE,cnt_iter)
        f1_validation, Y_labels_multi, y_predictions, rf=training_RF(new_train_set,train_labels, new_val_set, labels_val)
        print(f"Reached {f1_validation} F1 on validation set ")
        # Calcolare la matrice di confusione
       # conf_matrix = confusion_matrix(Y_labels_multi, y_predictions, labels=rf.classes_)
      #  disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf.classes_)
      #  disp.plot()
       # plt.show()
        return f1_validation


    
    #Selects n individuals to keep, as different as possible among individuals that have the most frequent modules as subtrees
    def get_individuals_to_keep(pop, n, modules_depth1,modules_depth2):
        
        sorted_modules_1 = dict(sorted(modules_depth1.items(), key=lambda x: x[1][0], reverse=True))
        sorted_modules_2 = dict(sorted(modules_depth2.items(), key=lambda x: x[1][0], reverse=True))
        # Estrapola i primi 5 elementi (quelli che si presentano di più) ... in questo modo potrò mantenere al massimo 20 individui, se vengono richiesti meno li seleziono in base alla fitness
        modules_freq = {}
        for key, value in list(sorted_modules_1.items())[:5]:
            value[1] /= value[0] #media delle fitness
            modules_freq[key] = value

        for key, value in list(sorted_modules_2.items())[:5]:
            value[1] /= value[0]
            modules_freq[key] = value
        
        # normalizzazione delle fitness
        somma_fitnesses_modules = sum(v[1] for v in modules_freq.values())
        for module in modules_freq:
            if somma_fitnesses_modules != 0:
                modules_freq[module][1] = modules_freq[module][1] / somma_fitnesses_modules
            else:
                modules_freq[module][1] = 0
        
        #view_hist_fitness_freq(modules_freq)
        sorted_modules_fitness = dict(sorted(modules_freq.items(), key=lambda x: x[1][1], reverse=True))
        
        individuals_to_keep = []
        for module, value in sorted_modules_fitness.items():
            if len(individuals_to_keep)<n:
                individuals_to_keep.append(module)
            else:
                break

        print("\nIndividuals to keep:")
        for i in range(len(individuals_to_keep)):
            print(f"{i}: {individuals_to_keep[i]}")

        return individuals_to_keep

    
    max_val=1.5e+100
    min_val=1.5e-100
    def mul(x, y):
        try:
            result = x * y
            if math.isnan(result):
                if (x>0 and y>0) or (x<0 and y<0):
                    return max_val
                else:
                    return min_val
            return result
        except:
            if (x>0 and y>0) or (x<0 and y<0):
                return max_val
            else:
                return min_val
            

    def protectedDiv(x, y):
        try:
            if y==0:
                return 1
            result = x / y
            if math.isnan(result):
                if (x>0 and y>0) or (x<0 and y<0):
                    return max_val
                else:
                    return min_val
            return result
        except:
            if (x>0 and y>0) or (x<0 and y<0):
                return max_val
            else:
                return min_val
   

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

    hof= [None] * N_ITERATIONS
    individuals_to_keep=[None]* N_ITERATIONS
    cntTree=0

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
    
    ind_best=0
    for cnt in range(N_ITERATIONS):
        global cnt_iter
        if cnt_iter>0:
            MAX_DEPTH+=2
        pop = toolbox.population(n=N_POPOLATION)
        
        if ind_best !=0:
            pop[random.randrange(N_POPOLATION)]=ind_best
        
        identity=gp.PrimitiveTree.from_string("ARG"+str(int(KERNEL_SIZE/2)),pset)
        identity=creator.Individual(identity)
        pop[random.randrange(N_POPOLATION)]=identity

        hof[cnt]=tools.HallOfFame(3)
        pop, log = eaSimple_elit(pop, toolbox, 0.5, 0.1, N_GENERATIONS, stats=mstats, halloffame=hof[cnt], verbose=True)

        print(f"Best individual: {hof[cnt][0]}")
        #valuto su training, validation e test set
        f1_testSet=evalTestSet(hof[cnt][0])
        f1_valSet=evalValidationSet(hof[cnt][0])
        validation_f1.append(f1_valSet)
        f1_score.append(f1_testSet)
        statistic.append(log)        

        modules_depth1,modules_depth2=get_modules(pop)   

        #grafici frequenze
        #view_hist1(modules_depth1)
        #view_hist2(modules_depth2)

        #mantengo gli individui in cui si presenta un modulo tra i più frequenti, dopo di che mantengo quelli più diversi tra loro
        individuals_to_keep[cnt] = get_individuals_to_keep(pop, N_IND_TO_KEEP, modules_depth1,modules_depth2)
        
         
        # Crea una funzione per sostituire il numero dell'ARG con il valore specifico
        def sostituisci(m):
            global cnt_arg
            if(cnt_arg<2):
                output = "ARG" + str(cnt_arg)
                cnt_arg += 1
            else:
                cnt_arg=0
                output = "ARG" + str(cnt_arg)
                cnt_arg += 1
            return output
        
        for i in range(len(individuals_to_keep[cnt])):  
            global cnt_arg
            cnt_arg=0
            individuals_to_keep[cnt][i] =re.sub(r'ARG\d+', sostituisci, individuals_to_keep[cnt][i] )
        print(individuals_to_keep[cnt])
        individuals_to_keep[cnt] = list(set(individuals_to_keep[cnt]))
        print(individuals_to_keep[cnt])
        
        ind_best=hof[cnt][0]
        #rendo le mie stringhe individui
        for i in range(len(individuals_to_keep[cnt])):
            individuals_to_keep[cnt][i] = gp.PrimitiveTree.from_string(individuals_to_keep[cnt][i], pset)
            individuals_to_keep[cnt][i] = creator.Individual(individuals_to_keep[cnt][i])
        cnt1=cntTree
        #aggiungo alle primitive i moduli da mantenere nell'iterazioen successiva
        for i in range(len(individuals_to_keep[cnt])):
            #se modulo profondità 2
            if(depth(str(individuals_to_keep[cnt][i]))==2):
                func = gp.compile(expr=individuals_to_keep[cnt][i], pset=new_pset_depth2)
                pset.addPrimitive(func, 4, name=f"execTree{cnt1}")
            #se modulo profondità 1
            if(depth(str(individuals_to_keep[cnt][i]))==1):
                func = gp.compile(expr=individuals_to_keep[cnt][i], pset=new_pset_depth1)
                pset.addPrimitive(func, 2, name=f"execTree{cnt1}")
            if(depth(str(individuals_to_keep[cnt][i]))==None):
                print("ERRORE MODULO NON DI PROFONDITà 1 O 2")
            cnt1+=1
        
        cnt_iter+=1
        
        for i in range(len(individuals_to_keep[cnt])):
            #se modulo profondità 2
            if(depth(str(individuals_to_keep[cnt][i]))==2):
                func = gp.compile(expr=individuals_to_keep[cnt][i], pset=new_pset_depth2)
            #se modulo profondità 1
            if(depth(str(individuals_to_keep[cnt][i]))==1):
                func = gp.compile(expr=individuals_to_keep[cnt][i], pset=new_pset_depth1)
            new_pset_depth1.addPrimitive(func, 2, name=f"execTree{cntTree}")
            new_pset_depth2.addPrimitive(func, 4, name=f"execTree{cntTree}")
            cntTree+=1

        
        
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"best_individual_{current_time}.pickle", "wb") as f:
        pickle.dump(hof[N_ITERATIONS-1][0], f)
    with open(f"pset_{current_time}.pkl", "wb") as p:
        dill.dump(pset, p)
    with open(f"efimcost_{current_time}.txt", "w") as r:
        r.write(str(const)+"\n")
        r.write(str(KERNEL_SIZE)+"\n")

    return validation_f1, f1_score, statistic

