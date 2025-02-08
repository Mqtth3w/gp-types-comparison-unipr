# initially developed by Aeranna Cella, reviewed by Matteo Gianvenuti
import datetime
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from genetic_programming import *

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
    
    running = True
    msginfo = outbox[0]
    # to sync
    event1 = threading.Event()
    event2 = threading.Event()
    event3 = threading.Event()

    # CellaMethod tree
    threaded_task(modularGP_CellaMethod, event1, outbox[1], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    # StefanoMethod list
    threaded_task(modularGP_StefanoMethod, event2, outbox[2], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)
    # classical GP
    threaded_task(classicalGP, event3, outbox[3], n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size)

    msginfo.config(text="Running...")
    # sync
    event1.wait()
    event2.wait()
    event3.wait()

    msginfo.config(text="Tasks completed")
    running = False
    
def modularGP_CellaMethod(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    event.set()
    pass

def modularGP_StefanoMethod(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    event.set()
    pass

def classicalGP(event, outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):
    outbox.delete(1.0, tk.END)
    event.set()
    pass

'''
def run_script(outbox, n_run, max_depth, generations, pop_size, iterations, inds_to_keep, kernel_size):

    list_f1_tot=[]
    # Ottenere l'ora attuale come oggetto datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"results_{current_time}.txt"
    
    # Esecuzione dello script n volte e salvataggio dei risultati in un file
    with open(filename, "a") as results_file:
    
        results_file.write("PARAMETRI IMPOSTATI: \nNumero di generazioni: "+ str(generations)+ "\nNumero di iterazione: "+ str(iterations)+ "\nProfondita' massima dell'albero: "+ str(max_depth)+ "\nIndividui da mantenere: "+ str(individual_to_keep)+ "\nNumero di run da eseguire: "+ str(run)+ "\nKernel size: "+ str(kernel_size)+ "\nPopulation size: "+ str(pop_size)+ "\nDataset: "+ str(file_path)+ "\n\n")
        results_file.close() #chiudo perchè potrebbe non visualizzarsi il grafico

    for i in range(run):
        #chiamo il mio codice di programmazione genetica che effettua classificazione binaria passando i parametri settati dall'utente
        f1_validation, f1_test, statistic= modularGP(run,max_depth,generations,iterations,individual_to_keep,file_path, kernel_size, pop_size, i)
        if(i==run-1):
            message_label.config(text="Algoritmo terminato, visualizzazione grafico:")
        else:
            message_label.config(text=f"La run {i} è stata completata. In esecuzione run {i+1} e visualizzazione grafico run {i}:")
        root.update()
        #al termine della run del programma visualizzo il grafico dei risultati su validation e training set
        graph(f1_test, f1_validation, i)
        #scrivo i risultati su file
        with open(filename, "a") as results_file:
            results_file.write("\n------------------------------------------------------------------------------")
            results_file.write(f"\nESECUZIONE RUN {i}\n\n") 

            for iter in range(int(iterations)):
                #stampo statistiche
                results_file.write(f"\nStatistiche iterazione {iter}:\n ")
                results_file.write(str(statistic[iter]))
                results_file.write("\n")

                #stampo risultati F1 su test set
                results_file.write(f"\n\nF1 on test set dell'iterazione {iter}: ")
                list_f1_tot.append(f1_test[iter])
                results_file.write(str(f1_test[iter]))
                results_file.write("\n")
            
        
            #riassunto dei risultati 
            results_file.write(f"\nRICAPITOLANDO\nF1 on test set della run {i}:\n")
            avg=float(0)
            for f in f1_test:
                avg+=float(f)
                results_file.write(str(f))
                results_file.write("\n")
            

            results_file.write(f"\n\nF1 on validation set della run {i}: \n")
            for f in f1_validation:
                results_file.write(str(f))
                results_file.write("\n")

            avg=avg/float(iterations)    
            results_file.write(f"\nF1 MEDIA DELL'ESECUZIONE DELLA RUN {i}: ")
            results_file.write(str(avg)+"\n\n")
            results_file.close()
            
    #risultato finale dato come media di tutte f1 delle varie run effettuate
    with open(filename, "a") as results_file:
        results_file.write("\n\n------------------------------------------------------------------------------")
        results_file.write(f"\nF1 MEDIA COMPLESSIVA DI TUTTE LE RUN:")
        avg=float(0)
        for f in list_f1_tot:
            avg+=float(f)
        avg=avg/(float(iterations)*float(run))
        results_file.write(str(avg)+'\n')
        results_file.close()
'''   

#per visualizzare il grafico ad ogni fine run
def graph(frame, f1_training, f1_validation, cnt):
    #elimino il grafico precendente
    for widget in frame.grid_slaves():
        if int(widget.grid_info()["row"]) == 5 and int(widget.grid_info()["column"]) == 0:
            widget.grid_forget()
    x1=[]
    x2=[]
    for i in range(len(f1_training)):
        x1.append(i+1)
    for i in range(len(f1_validation)):
        x2.append(i+1)
    #creo il grafico a dispersione, uso anche plot per avere linea che congiunge
    fig = Figure(figsize=(7, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(x1, f1_training, color='blue', label='TEST SET')
    ax.scatter(x2, f1_validation, color='red', label='VALIDATION SET')
    ax.plot(x1, f1_training, color='blue')
    ax.plot(x2, f1_validation, color='red')
    ax.set_xlabel('ITERAZIONI')
    ax.set_ylabel('F1')
    ax.set_title(f'Validation and test set F1, run numero {cnt}')
    ax.legend()
    #creo un oggetto FigureCanvasTkAgg che contiene il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, padx=20, pady=10)

running = False
def ui():
    root = tk.Tk()
    root.title("GP types comparsion")
    root.resizable(False, False)

    frame = tk.Frame(root)
    frame.pack()

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

    dataset_into_frame = tk.LabelFrame(frame, text="Insert already formatted dataset label-target for binary classification:")
    dataset_into_frame.grid(row=1, column=0, padx=20, pady=20, sticky="news")

    # upload csv file (dataset)
    button = tk.Button(dataset_into_frame, text="Upload csv dataset", command=upload_csv)
    button.grid(row=0, column=0, padx=50, pady=20)
    '''
    target = tk.Label(dataset_into_frame, text="Label target:")
    target.grid(row=0,column=1, pady=20)
    target_entry = tk.Entry(dataset_into_frame)
    target_entry.grid(row=0, column=2, pady=20)
    '''
    # run button 
    button = tk.Button(frame, text="Run", command=lambda: threaded_task(start_tasks, None, [message_label, outbox1, outbox2, outbox3],
                                                                      int(n_run_spinbox.get()), int(max_depth_spinbox.get()), 
                                                                      int(selected_option_generation.get()), int(selected_option_popolation.get()), 
                                                                      int(selected_option_iteration.get()), int(ind_to_keep_spinbox.get()),
                                                                      int(kernel_size_option.get())))
    button.grid(row=3, column=0, sticky="news", padx=20, pady=10)
    
    # create a label for the info message
    message_label = tk.Label(frame, text="General output info here")
    message_label.grid(row=4, column=0, sticky="news", padx=20, pady=10)

    # show three output box in //
    output_frame = tk.Frame(frame)
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

    root.mainloop()



if __name__ == '__main__':
    ui()