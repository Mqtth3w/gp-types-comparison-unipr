import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure

from GPmodular import modularGP

#per caricare file CSV
file_path=""
def upload_csv():
    global file_path
    file_path = filedialog.askopenfilename()
    print(file_path)
    

#per visualizzare il grafico ad ogni fine run
def graph(f1_training,f1_validation,cnt):
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
    #aggiorno grafica
    window.update()


def run_script():
    if(len(n_run_entry.get())>0 and len(kernel_entry.get())>0):# and len(target_entry.get())>0):
        # Recupero dei valori inseriti dall'utente
        run=int(n_run_entry.get())
        max_depth=max_depth_spinbox.get()
        generations=selected_option_generation.get()
        pop_size=selected_option_popolation.get()
        iterations=selected_option_iteration.get()
        individual_to_keep=ind_to_keep_spinbox.get()
        kernel_size=kernel_entry.get()
        global file_path

        if(len(file_path)>0):
            message_label.config(text="Algoritmo in esecuzione...")
            window.update()

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
                window.update()
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
        else:
           tk.messagebox.showwarning(title="Error",message="Inserisci file csv")
 
    else:
        tk.messagebox.showwarning(title="Error",message="Inserisci tutti i parametri")



window=tk.Tk()
window.title("RUN AUTOMATICHE")

frame=tk.Frame(window)
frame.pack()

param_into_frame=tk.LabelFrame(frame, text="Inserisci parametri per la run")
param_into_frame.grid(row=0, column=0, padx=20, pady=20)

#parametri
n_run= tk.Label(param_into_frame, text="Numero di run")
n_run.grid(row=0,column=0, padx=50,pady=10)
n_run_entry=tk.Entry(param_into_frame)
n_run_entry.grid(row=1,column=0, padx=50)

ind_to_keep= tk.Label(param_into_frame, text="Individui da mantenere")
ind_to_keep.grid(row=0,column=1,  padx=50, pady=10)
ind_to_keep_spinbox=tk.Spinbox(param_into_frame, from_=1, to =10)
ind_to_keep_spinbox.grid(row=1,column=1,  padx=50)

max_depth= tk.Label(param_into_frame, text="Max depth")
max_depth.grid(row=0,column=2, padx=50, pady=10)
max_depth_spinbox=tk.Spinbox(param_into_frame, from_=4, to =10)
max_depth_spinbox.grid(row=1,column=2, padx=50)

options = [i+1 for i in range(10)]
selected_option_iteration = tk.StringVar(param_into_frame)
selected_option_iteration.set(options[0])  # Opzione di default
n_iterations = tk.Label(param_into_frame, text="Numero di iterazioni per run")
n_iterations_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_iteration)
n_iterations.grid(row=2, column=0, padx=50, pady=10)
n_iterations_combobox.grid(row=3, column=0, padx=50)

options = [i+1 for i in range(100)]
selected_option_generation= tk.StringVar(param_into_frame)
selected_option_generation.set(options[0])  # Opzione di default
n_generations = tk.Label(param_into_frame, text="Numero di generazioni per run")
n_generations_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_generation)
n_generations.grid(row=2, column=1, padx=50, pady=10)
n_generations_combobox.grid(row=3, column=1 , padx=50)

options = [i+1 for i in range(19,1000)]
selected_option_popolation= tk.StringVar(param_into_frame)
selected_option_popolation.set(options[30])  # Opzione di default
n_popolation = tk.Label(param_into_frame, text="Dimensione della popolazione")
n_popolation_combobox = ttk.Combobox(param_into_frame, values=options, textvariable=selected_option_popolation)
n_popolation.grid(row=4, column=0, padx=50, pady=10)
n_popolation_combobox.grid(row=5, column=0 , padx=50)

kernel= tk.Label(param_into_frame, text="Dimensione kernel")
kernel.grid(row=2,column=2, padx=50,pady=10)
kernel_entry=tk.Entry(param_into_frame)
kernel_entry.grid(row=3,column=2, padx=50)


dataset_into_frame=tk.LabelFrame(frame, text="Inserisci dataset già formattato e label target per classificazione binaria:")
dataset_into_frame.grid(row=1, column=0, padx=20, pady=20, sticky="news")

# Button dataset
button = tk.Button(dataset_into_frame, text="Carica dataset CSV", command=upload_csv)
button.grid(row=0, column=0, padx=50, pady=20)
'''
target= tk.Label(dataset_into_frame, text="Label target:")
target.grid(row=0,column=1,pady=20)
target_entry=tk.Entry(dataset_into_frame)
target_entry.grid(row=0,column=2, pady=20)
'''

# Button Esegui
button = tk.Button(frame, text="Esegui", command=run_script)
button.grid(row=3, column=0, sticky="news", padx=20, pady=10)
 
# Crea un'etichetta per il messaggio
message_label = tk.Label(frame, text="")
message_label.grid(row=4, column=0, sticky="news", padx=20, pady=10)

window.mainloop()