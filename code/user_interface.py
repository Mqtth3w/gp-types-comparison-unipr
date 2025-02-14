# initially developed by Aeranna Cella, reviewed by Francesca Stefano and afterwards by Matteo Gianvenuti
'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import multiprocessing
from gp_types import *

def threaded_task(task_func, task_func2, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    multiprocessing.Process(
        target=task_func,
        args=(task_func2, file_path, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size),
        daemon=True
    ).start()
    
file_path = ""
def upload_csv():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    
def start_task(task_func, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    # params check
    if not file_path or not os.path.isfile(file_path):
        messagebox.showerror("Error", "Select a csv dataset file.")
        return
    threaded_task(run_script, task_func, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)

def run_script(method_func, file_path, verbose, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    avg_f1 = []
    current_time = datetime.datetime.now()
    start = current_time.timestamp()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"{script_dir}/{method_func.__name__}_results_{current_time}.txt"
    print(f"({method_func.__name__}) outfile {filename}")
    # run the script n times and save the results to a file
    with open(filename, "w") as results_file0:
        results_file0.write("runs;max_depth;generations;population_size;iterations;individuals_to_keep;kernel_size;method;dataset\n")
        results_file0.write(f"{n_run}:{max_depth};{generations};{pop_size};{iterations};{inds_to_keep};{kernel_size};{method_func.__name__};{file_path}\n")
    for i in range(n_run):
        print(f"({method_func.__name__}) Run {i} starting...")
        run_start = datetime.datetime.now().timestamp()
        _, num_nodes, f1_validation, f1_test, _ = method_func(current_time, file_path, verbose, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size, i)
        run_end = datetime.datetime.now().timestamp()
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

    end = datetime.datetime.now().timestamp()
    # final result given as the average of all f1s of the various runs carried out
    avg = sum(map(float, avg_f1)) / n_run #len(avg_f1)
    with open(filename, "a") as results_file2:
        results_file2.write("\n--------------------\n--------------------\n--------------------\n")
        results_file2.write(f"F1_OVERALL_AVERAGE_OF_ALL_RUNS;OVERALL_RUNNING_TIME_MIN\n")
        results_file2.write(f"{avg};{(end - start) / 60}\n")
    print(f"({method_func.__name__}) finished, took {(end - start) / 60} minutes")

def ui():
    root = tk.Tk()
    root.title("GP types comparison")
    root.geometry("1100x450") #WxH

    frame=tk.Frame(root)
    frame.pack(fill="both", expand=True)

    param_into_frame = tk.LabelFrame(frame, text="Insert parameters")
    param_into_frame.grid(row=0, column=0, padx=20, pady=10, sticky="news")

    # parameters
    n_run = tk.Label(param_into_frame, text="Number of runs")
    n_run.grid(row=0, column=0, padx=50, pady=10)
    n_run_spinbox = tk.Spinbox(param_into_frame, from_=1, to=5)
    n_run_spinbox.grid(row=1, column=0, padx=50)

    ind_to_keep = tk.Label(param_into_frame, text="Individuals to keep")
    ind_to_keep.grid(row=0, column=1,  padx=50, pady=10)
    ind_to_keep_spinbox = tk.Spinbox(param_into_frame, from_=1, to=10)
    ind_to_keep_spinbox.grid(row=1, column=1,  padx=50)

    max_depth = tk.Label(param_into_frame, text="Max depth")
    max_depth.grid(row=0, column=2, padx=50, pady=10)
    max_depth_spinbox = tk.Spinbox(param_into_frame, from_=4, to=10)
    max_depth_spinbox.grid(row=1, column=2, padx=50)

    options = list(range(1, 11))
    selected_option_iteration = tk.StringVar(param_into_frame)
    selected_option_iteration.set(options[0])  # default option
    n_iterations = tk.Label(param_into_frame, text="Number of iterations per run")
    n_iterations_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_iteration)
    n_iterations.grid(row=2, column=0, padx=50, pady=10)
    n_iterations_combobox.grid(row=3, column=0, padx=50)

    options = list(range(1, 101))
    selected_option_generation = tk.StringVar(param_into_frame)
    selected_option_generation.set(options[0])  # default option
    n_generations = tk.Label(param_into_frame, text="Number of generations per iter")
    n_generations_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_generation)
    n_generations.grid(row=2, column=1, padx=50, pady=10)
    n_generations_combobox.grid(row=3, column=1 , padx=50)

    options = list(range(20, 1001))
    selected_option_popolation = tk.StringVar(param_into_frame)
    selected_option_popolation.set(options[0])  # default option
    n_popolation = tk.Label(param_into_frame, text="Population size")
    n_popolation_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_popolation)
    n_popolation.grid(row=0, column=3, padx=50, pady=10)
    n_popolation_combobox.grid(row=1, column=3, padx=50)

    options = list(range(3, 13, 2))
    kernel_size_option = tk.StringVar(param_into_frame)
    kernel_size_option.set(options[0]) # default option
    kernel = tk.Label(param_into_frame, text="Kernel size")
    kernel_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=kernel_size_option)
    kernel.grid(row=2, column=2, padx=50, pady=10)
    kernel_combobox.grid(row=3, column=2, padx=50, pady=10)
    
    options = ["False", "True"]
    verbose_option = tk.StringVar(param_into_frame)
    verbose_option.set(options[1]) # default option
    verbose = tk.Label(param_into_frame, text="Verbose")
    verbose_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=verbose_option)
    verbose.grid(row=2, column=3, padx=50, pady=10)
    verbose_combobox.grid(row=3, column=3, padx=50, pady=10)

    dataset_into_frame = tk.LabelFrame(frame, text="Insert already formatted dataset label-target for binary classification:")
    dataset_into_frame.grid(row=1, column=0, padx=20, pady=20, sticky="news")

    # upload csv file (dataset)
    button = ttk.Button(dataset_into_frame, text="Upload csv dataset", command=upload_csv)
    button.grid(row=0, column=0, padx=50, pady=20)
    
    # run buttons
    buttons_frame = tk.LabelFrame(frame, text="Run method")
    buttons_frame.grid(row=2, column=0, padx=20, pady=20, sticky="news")
    # modularGP_CellaMethod
    button1 = ttk.Button(buttons_frame, text="Run  modularGP_CellaMethod", command=lambda: start_task(modularGP_CellaMethod, verbose_option.get() == "True",
                                                                                 int(n_run_spinbox.get()), int(max_depth_spinbox.get()), 
                                                                                 int(selected_option_generation.get()), int(selected_option_popolation.get()), 
                                                                                 int(selected_option_iteration.get()), int(ind_to_keep_spinbox.get()),
                                                                                 int(kernel_size_option.get())))
    button1.grid(row=0, column=0, sticky="news", padx=20, pady=10)
    # modularGP_StefanoMethod
    button2 = ttk.Button(buttons_frame, text="Run  modularGP_StefanoMethod", command=lambda: start_task(modularGP_StefanoMethod, verbose_option.get() == "True",
                                                                                 int(n_run_spinbox.get()), int(max_depth_spinbox.get()), 
                                                                                 int(selected_option_generation.get()), int(selected_option_popolation.get()), 
                                                                                 int(selected_option_iteration.get()), int(ind_to_keep_spinbox.get()),
                                                                                 int(kernel_size_option.get())))
    button2.grid(row=0, column=1, sticky="news", padx=20, pady=10)
    # classicalGP
    button3 = ttk.Button(buttons_frame, text="Run  classicalGP", command=lambda: start_task(classicalGP, verbose_option.get() == "True",
                                                                                 int(n_run_spinbox.get()), int(max_depth_spinbox.get()), 
                                                                                 int(selected_option_generation.get()), int(selected_option_popolation.get()), 
                                                                                 int(selected_option_iteration.get()), int(ind_to_keep_spinbox.get()),
                                                                                 int(kernel_size_option.get())))
    button3.grid(row=0, column=2, sticky="news", padx=20, pady=10)
    
    root.mainloop()
    
if __name__ == '__main__':
    ui()