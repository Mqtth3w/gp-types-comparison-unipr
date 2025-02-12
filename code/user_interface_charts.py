'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def parse_results_file(filename):
    parameters = {}
    runs = []
    overall = {}
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        
        if len(lines) < 2:
            return parameters, runs, overall
        
        params_line = lines[1].split(';')
        parameters = {
            'n_run': params_line[0],
            'max_depth': params_line[1],
            'generations': params_line[2],
            'population_size': params_line[3],
            'iterations': params_line[4],
            'individuals_to_keep': params_line[5],
            'kernel_size': params_line[6],
            'dataset': params_line[7]
        }
        
        current_run = None
        line_index = 2
        
        while line_index < len(lines):
            line = lines[line_index]
            
            # start of a new run's iteration data
            if line == "run;iteration;test_f1;validation_f1":
                current_run = {
                    'test_f1': [],
                    'validation_f1': [],
                    'avg_test_f1': None,
                    'num_nodes_best_ind': None,
                    'run_time_min': None
                }
                runs.append(current_run)
                line_index += 1
                
                # read iteration data lines
                while line_index < len(lines):
                    current_line = lines[line_index]
                    
                    # stop at section break or summary header
                    if current_line.startswith("----------") or current_line.startswith("run;avg_test_f1"):
                        break
                    
                    # process valid data lines
                    if ";" in current_line:
                        parts = current_line.split(";")
                        if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
                            try:
                                current_run['test_f1'].append(float(parts[2]))
                                current_run['validation_f1'].append(float(parts[3]))
                            except (ValueError, IndexError):
                                print(f"Error processing iterations, line_index {line_index}")
                    line_index += 1
            
            # run summary section
            elif line.startswith("run;avg_test_f1"):
                line_index += 1
                if line_index < len(lines) and runs:
                    parts = lines[line_index].split(";")
                    if len(parts) >= 4:
                        try:
                            runs[-1]['avg_test_f1'] = float(parts[1])
                            runs[-1]['num_nodes_best_ind'] = int(parts[2])
                            runs[-1]['run_time_min'] = float(parts[3])
                        except (ValueError, IndexError):
                            print(f"Error run summary section, line_index {line_index}")
                line_index += 1
            
            # overall results section
            elif line.startswith("F1_OVERALL"):
                line_index += 1
                if line_index < len(lines):
                    parts = lines[line_index].split(";")
                    if len(parts) >= 2:
                        try:
                            overall['F1_OVERALL_AVERAGE_OF_ALL_RUNS'] = float(parts[0])
                            overall['OVERALL_RUNNING_TIME_MIN'] = float(parts[1])
                        except (ValueError, IndexError):
                            print(f"Error overall section, line_index {line_index}")
                break
             
            else:
                line_index += 1
                   
    return parameters, runs, overall

def load_results():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not filepath:
        return
    
    parameters, runs, overall = parse_results_file(filepath)
    if not parameters or not runs or not overall:
        messagebox.showerror("Error", "Select a valid results file file.")
        return
    
    # results window
    results_window = tk.Toplevel(root)
    results_window.title("Analysis Results")
    results_window.geometry("1200x800")
    
    # create a canvas widget to make the whole UI scrollable
    canvas = tk.Canvas(results_window)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_mouse_wheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    scrollable_frame.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
    
    # display parameters
    params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters")
    params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    # CHARTS
    row = 0
    for param, value in parameters.items():
        label_text = f"{param.replace('_', ' ').title()}: {value}"
        ttk.Label(params_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        row += 1
    
    # display runs
    for run_idx, run_data in enumerate(runs):
        run_frame = ttk.LabelFrame(scrollable_frame, text=f"run {run_idx+1}")
        run_frame.grid(row=run_idx+1, column=0, padx=10, pady=10, sticky="nsew")
        
        # create two columns: left for chart, right for metrics
        chart_frame = ttk.Frame(run_frame)
        metrics_frame = ttk.Frame(run_frame)
        
        chart_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        metrics_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # create chart
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        iterations = list(range(1, len(run_data['test_f1']) + 1))
        ax.plot(iterations, run_data['test_f1'], 'b-o', label='Test F1')
        ax.plot(iterations, run_data['validation_f1'], 'r--s', label='Validation F1')
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 Score progression - run {run_idx+1}")
        ax.legend()
        ax.grid(True)
        
        chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # add metrics
        metrics_data = [
            ("Average Test F1:", f"{run_data['avg_test_f1']:.4f}"),
            ("Run Time (min):", f"{run_data['run_time_min']:.2f}"),
            ("Best Individual Nodes:", run_data['num_nodes_best_ind'])
        ]
        
        for i, (label, value) in enumerate(metrics_data):
            ttk.Label(metrics_frame, text=label, font=('Arial', 10, 'bold'))\
                .grid(row=i, column=0, sticky="e", padx=5, pady=5)
            ttk.Label(metrics_frame, text=value)\
                .grid(row=i, column=1, sticky="w", padx=5, pady=5)
    
    # display overall results
    overall_frame = ttk.LabelFrame(scrollable_frame, text="Overall Results")
    overall_frame.grid(row=len(runs)+1, column=0, padx=10, pady=20, sticky="ew")
    
    overall_data = [
        ("Overall Average F1:", f"{overall['F1_OVERALL_AVERAGE_OF_ALL_RUNS']:.4f}"),
        ("Total Running Time (min):", f"{overall['OVERALL_RUNNING_TIME_MIN']:.2f}")
    ]
    
    for i, (label, value) in enumerate(overall_data):
        ttk.Label(overall_frame, text=label, font=('Arial', 10, 'bold'))\
            .grid(row=i, column=0, sticky="e", padx=10, pady=5)
        ttk.Label(overall_frame, text=value)\
            .grid(row=i, column=1, sticky="w", padx=10, pady=5)

def ui():
    global root
    root = tk.Tk()
    root.title("Results Analysis")
    root.geometry("900x150")
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)
    
    upload_frame = ttk.LabelFrame(main_frame, text="Upload a results (*.txt) file to do Results Analysis. \
                                  You can upload multiple files (one after the other) to compare them in parallel")
    upload_frame.pack(pady=20, padx=20, fill="x")
    
    ttk.Button(upload_frame, text="Upload Results File", command=load_results).pack(pady=15, padx=50)
    
    root.mainloop()

if __name__ == '__main__':
    ui()