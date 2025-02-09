# initially developed by Aeranna Cella, reviewed by Matteo Gianvenuti
from datetime import datetime
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gp_types import *

def threaded_task(task_func, event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    thread = threading.Thread(
        target=task_func,
        args=(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size),
        daemon=True
    )
    thread.start()

file_path = ""
def upload_csv():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    if not os.path.isfile(file_path): # should never happen
        messagebox.showerror("Error", "Select only a csv file.")
    
def start_tasks(_, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    # run in // the three types of GP
    global running
    if running:
        messagebox.showerror("Error", "The previous run must complete before another can be run.")
        return
    
    # params check
    if not(file_path and os.path.isfile(file_path)):
        messagebox.showerror("Error", "Select a csv dataset file.")
        return
    
    running = True # monitor
    msginfo = outbox[0]
    # to sync
    event1 = threading.Event()
    event2 = threading.Event()
    event3 = threading.Event()

    # CellaMethod tree
    threaded_task(modularGP_CellaMethod, event1, [outbox[1], outbox[4], 0], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    # StefanoMethod list
    threaded_task(modularGP_StefanoMethod, event2, [outbox[2], outbox[4], 1], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    # classical GP
    threaded_task(classicalGP, event3, [outbox[3], outbox[4], 2], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)

    msginfo.config(text="Running...")
    # sync
    event1.wait()
    event2.wait()
    event3.wait()

    msginfo.config(text="Tasks completed")
    running = False
    
def start_modularGP_CellaMethod(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    run_script(modularGP_CellaMethod, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    event.set()

def strat_modularGP_StefanoMethod(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    run_script(modularGP_StefanoMethod, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    event.set()

def start_classicalGP(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    run_script(classicalGP, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    event.set()

def run_script(method_func, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    avg_f1 = []
    current_time = datetime.datetime.now()
    start = current_time.timestamp()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{method_func.__name__}_results_{current_time}.txt"
    outbox_col = outbox[2]
    frame = outbox[1]
    outbox = outbox[0]
    
    # run the script n times and save the results to a file
    with open(filename, "w") as results_file:
        results_file.write(f"Parameters set:\ngenerations: {generations}\niterations: {iterations}\nmax depth: {max_depth}\nindividuals to keep: {inds_to_keep}\nnumber of runs: {n_run}\nkernel size: {kernel_size}\npopulation size: {pop_size}\ndataset: {file_path}\n\n")

    for i in range(n_run):
        f1_validation, f1_test, statistic = method_func(file_path, outbox, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size, i)
        if i == n_run-1:
            outbox.insert(tk.END, "Algorithm finished, charts view:")
        else:
            outbox.insert(tk.END, f"Run {i} completed. Run {i+1} started. Charts view of run {i}:")
        
        # at the end of the program run display the charts of the results on validation and training set
        graph(frame, outbox_col, f1_test, f1_validation, i)
        # save results
        with open(filename, "a") as results_file:
            results_file.write("\n------------------------------------------------------------------------------")
            results_file.write(f"\nRUN {i}\n") 

            for iter in range(int(iterations)):
                results_file.write(f"Iteration {iter} Statistics: {statistic[iter]}\n")
                results_file.write(f"F1 on test set of iter {iter}: {f1_test[iter]}\n")
                
            # recap 
            results_file.write(f"\nRECAP\nF1 on test set of run {i}:\n")
            avg = sum(map(float, f1_test)) / iterations #len(f1_test)
            avg_f1.append(avg)
            results_file.writelines(f"{f}\n" for f in f1_test)            

            results_file.write(f"\n\nF1 on validation set of run {i}:\n")
            results_file.writelines(f"{f}\n" for f in f1_validation)

            results_file.write(f"\nMEAN F1 OF RUN {i}: {avg}\n\n")

    end = datetime.datetime.now().timestamp()
    # final result given as the average of all f1s of the various runs carried out
    avg = sum(map(float, avg_f1)) / n_run #len(avg_f1)
    with open(filename, "a") as results_file:
        results_file.write("\n\n------------------------------------------------------------------------------")
        results_file.write(f"\nF1 OVERALL AVERAGE OF ALL RUNS: {avg}\nRUNNING TIME: {(end - start) / 60} minutes")

# to display the chart at the end of each run
def graph(frame, outbox_col, f1_training, f1_validation, n_run):
    chart_row = 2
    # remove the previous chart for the specific outbox column
    for widget in frame.grid_slaves():
        # Check if the widget is in the correct column and row for the chart
        if int(widget.grid_info().get("row", 0)) == chart_row and int(widget.grid_info().get("column", 0)) == outbox_col:
            widget.grid_forget()  # Remove only the chart in the specific outbox position

    x1 = list(range(1, len(f1_training) + 1))
    x2 = list(range(1, len(f1_validation) + 1))

    # Create the figure for the graph
    fig = plt.Figure(figsize=(5, 4), dpi=90)
    ax = fig.add_subplot(111)

    # Plot the data
    ax.scatter(x1, f1_training, color='blue', label='TEST SET')
    ax.scatter(x2, f1_validation, color='red', label='VALIDATION SET')
    ax.plot(x1, f1_training, color='blue')
    ax.plot(x2, f1_validation, color='red')
    ax.set_xlabel('ITERATIONS')
    ax.set_ylabel('F1')
    ax.set_title(f'Validation and Test Set F1, Run Number: {n_run}')
    ax.legend()

    # Create the canvas and add it to the grid in the correct position (directly below the outbox)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()

    # Place the chart directly below the corresponding outbox
    canvas.get_tk_widget().grid(row=chart_row, column=outbox_col, padx=20, pady=10)

running = False
def ui():
    root = tk.Tk()
    root.title("GP types comparsion")
    root.geometry("2000x700")

    # create a Canvas widget to make the whole UI scrollable
    canvas = tk.Canvas(root)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_frame_configure(_):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_mouse_wheel(event):
        if event.delta > 0:
            canvas.yview_scroll(-1, "units") # up
        else:
            canvas.yview_scroll(1, "units") # down

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    scrollable_frame.bind("<Configure>", on_frame_configure)

    param_into_frame = tk.LabelFrame(scrollable_frame, text="Insert parameters")
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
    n_generations = tk.Label(param_into_frame, text="Number of generations per run")
    n_generations_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_generation)
    n_generations.grid(row=2, column=1, padx=50, pady=10)
    n_generations_combobox.grid(row=3, column=1 , padx=50)

    options = list(range(20, 1001))  
    selected_option_popolation = tk.StringVar(param_into_frame)
    selected_option_popolation.set(options[30])  # default option
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

    dataset_into_frame = tk.LabelFrame(scrollable_frame, text="Insert already formatted dataset label-target for binary classification:")
    dataset_into_frame.grid(row=1, column=0, padx=20, pady=20, sticky="news")

    # upload csv file (dataset)
    button = tk.Button(dataset_into_frame, text="Upload csv dataset", command=upload_csv)
    button.grid(row=0, column=0, padx=50, pady=20)
    
    # run button 
    button = tk.Button(scrollable_frame, text="Run", command=lambda: threaded_task(start_tasks, None, [message_label, outbox1, outbox2, outbox3, output_frame],
                                                                      int(n_run_spinbox.get()), int(max_depth_spinbox.get()), 
                                                                      int(selected_option_generation.get()), int(selected_option_popolation.get()), 
                                                                      int(selected_option_iteration.get()), int(ind_to_keep_spinbox.get()),
                                                                      int(kernel_size_option.get())))
    button.grid(row=3, column=0, sticky="news", padx=20, pady=10)
    
    # create a label for the info message
    message_label = tk.Label(scrollable_frame, text="General output info here")
    message_label.grid(row=4, column=0, sticky="news", padx=20, pady=10)

    # show three output boxes in //
    output_frame = tk.Frame(scrollable_frame)
    output_frame.grid(row=5, column=0, padx=20, pady=10, sticky="news")
    output_frame.columnconfigure(0, weight=2)
    output_frame.columnconfigure(1, weight=2)
    output_frame.columnconfigure(2, weight=2)
    tk.Label(output_frame, text="modularGP_CellaMethod").grid(row=0, column=0, pady=(0, 5))
    tk.Label(output_frame, text="modularGP_StefanoMethod").grid(row=0, column=1, pady=(0, 5))
    tk.Label(output_frame, text="classicalGP").grid(row=0, column=2, pady=(0, 5))
    # modularGP_CellaMethod
    outbox1 = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=55, height=20)
    outbox1.grid(row=1, column=0, padx=5, pady=5, sticky="news")
    # modularGP_StefanoMethod
    outbox2 = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=55, height=20)
    outbox2.grid(row=1, column=1, padx=5, pady=5, sticky="news")
    # classicalGP
    outbox3 = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=55, height=20)
    outbox3.grid(row=1, column=2, padx=5, pady=5, sticky="news")
    #graph(output_frame, list(range(1, 1100, 2)), list(range(1, 1010)), 1, 1)
    #graph(output_frame, list(x*x for x in range(1, 2200, 1)), list(x*x for x in range(1, 2020, 2)), 1, 2)
    #graph(output_frame, list(x*x*2 for x in range(1, 3030, 2)), list(x*x*2 for x in range(1, 3030, 3)), 1, 3)
    root.mainloop()

    
if __name__ == '__main__':
    ui()