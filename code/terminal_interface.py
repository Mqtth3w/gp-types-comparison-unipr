# initially developed by Aeranna Cella, reviewed by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
from datetime import datetime
from gp_types import *
import argparse
import os

def run_script(method_func, file_path, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    avg_f1 = []
    current_time = datetime.now()
    start = current_time.timestamp()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"{script_dir}/{method_func.__name__}_results_{current_time}.txt"
    print(f"({method_func.__name__}) outfile {filename}")
    if method_func.__name__ == "classicalGP":
        n_run = 1
        iterations = 1
        inds_to_keep = 0
    # run the script n times and save the results to a file
    with open(filename, "w") as results_file0:
        results_file0.write("runs;max_depth;generations;population_size;iterations;individuals_to_keep;kernel_size;method;dataset\n")
        results_file0.write(f"{n_run};{max_depth};{generations};{pop_size};{iterations};{inds_to_keep};{kernel_size};{method_func.__name__};{file_path}\n")
    for i in range(n_run):
        print(f"({method_func.__name__}) Run {i} starting...")
        run_start = datetime.now().timestamp()
        _, num_nodes, f1_validation, f1_test, _ = method_func(current_time, file_path, verbose, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size, i)
        run_end = datetime.now().timestamp()
        print(f"({method_func.__name__}) Run {i} completed, run time {(run_end - run_start) / 60} minutes")
        
        # save results
        with open(filename, "a") as results_file:
            results_file.write("\n--------------------\n--------------------\n")
            results_file.write(f"run;iteration;test_f1;validation_f1\n") 

            for iter in range(int(iterations)):
                results_file.write(f"{i};{iter};{f1_test[iter]};{f1_validation[iter]}\n")
                
            # avg f1 of the run
            avg = sum(map(float, f1_test)) / iterations #len(f1_test)
            avg_f1.append(avg)

            results_file.write("\n--------------------\n")
            results_file.write(f"run;avg_test_f1;num_nodes_best_ind;run_time_min\n{i};{avg};{num_nodes};{(run_end - run_start) / 60}\n")

    end = datetime.now().timestamp()
    # final result given as the average of all f1s of the various runs carried out
    avg = sum(map(float, avg_f1)) / n_run #len(avg_f1)
    with open(filename, "a") as results_file2:
        results_file2.write("\n--------------------\n--------------------\n--------------------\n")
        results_file2.write(f"F1_OVERALL_AVERAGE_OF_ALL_RUNS;OVERALL_RUNNING_TIME_MIN\n")
        results_file2.write(f"{avg};{(end - start) / 60}\n")
    print(f"({method_func.__name__}) finished, took {(end - start) / 60} minutes")


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--runs', type=int, default=1, choices=list(range(1, 11)), help='Number of runs')
    parser.add_argument('--max_depth', type=int, default=4 , choices=list(range(4, 11)), help='Tree max depth')
    parser.add_argument('--generations', type=int, default=5, choices=list(range(1, 101)), help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=50, choices=list(range(20, 1001)), help='Number of individuals (population)')
    parser.add_argument('--iterations', type=int, default=3, choices=list(range(1, 11)), help='Number of iterations')
    parser.add_argument('--inds_to_keep', type=int, default=3, choices=list(range(1, 11)), help='Number of individuals to be kept')
    parser.add_argument('--kernel_size', type=int, default=3, choices=list(range(3, 13, 2)), help='Kernel size')
    parser.add_argument('--verbose', type=bool, default=True, choices=[True, False], help='Verbose')
    parser.add_argument('--method', type=str, default='modularGPCella', choices=['modularGPCella', 'modularGPStefano', 'classicalGP'], help='GP method')
    parser.add_argument('--dataset', type=str, default='../dataset/arrhythmias_data_balanced_3000elements.csv', helps='Datset file path ex: ./data/arrhythmias.csv')
    
    return parser.parse_args()

def main(args):
    my_dict = {'modularGPCella': modularGP_CellaMethod, 
               'modularGPStefano': modularGP_StefanoMethod, 
               'classicalGP': classicalGP}
    run_script(my_dict[args.method], args.dataset, args.verbose, args.runs, args.max_depth, 
               args.generations, args.pop_size, args.iterations, args.inds_to_keep, args.kernel_size)
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)