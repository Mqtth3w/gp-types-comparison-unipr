# initially developed by Aeranna Cella, reviewed by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
from gp_types import *
from user_interface import run_script
import argparse

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
    parser.add_argument('--method', type=str, default='modularGPStefano', choices=['modularGPCella', 'modularGPStefano', 'classicalGP'], help='GP method')
    parser.add_argument('--dataset', type=str, default='../dataset/arrhythmias_data_balanced_3000elements.csv', help='Datset file path ex: ./data/arrhythmias.csv')
    
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