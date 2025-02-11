'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

def parse_results_file(filename):
    parameters = {}
    runs = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 2:
            return None, None
        
        # Parse parameters line (second line)
        params_line = lines[1].strip().split(';')
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
        
        # Parse runs data
        current_run = None
        for line in lines[2:]:
            stripped = line.strip()
            if stripped.startswith('run;iteration;test_f1;validation_f1'):
                current_run = {'test_f1': [], 'validation_f1': []}
                runs.append(current_run)
            elif current_run is not None and ';' in stripped and not stripped.startswith('----'):
                parts = stripped.split(';')
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        current_run['test_f1'].append(float(parts[2]))
                        current_run['validation_f1'].append(float(parts[3]))
                    except (ValueError, IndexError):
                        pass
    return parameters, runs

def load_results():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not filepath:
        return
    
    parameters, runs = parse_results_file(filepath)
    if not parameters or not runs:
        return
    
    # Create results window
    results_window = tk.Toplevel(root)
    results_window.title("Analysis Results")
    results_window.geometry("1200x800")
    
    # Create scrollable canvas
    canvas = tk.Canvas(results_window)
    scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Display parameters
    params_frame = ttk.LabelFrame(scrollable_frame, text="Experiment Parameters")
    params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    
    row = 0
    for param, value in parameters.items():
        label_text = f"{param.replace('_', ' ').title()}: {value}"
        ttk.Label(params_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        row += 1
    
    # Display charts
    for run_idx, run_data in enumerate(runs):
        chart_frame = ttk.Frame(scrollable_frame)
        chart_frame.grid(row=run_idx+1, column=0, padx=10, pady=20, sticky="nsew")
        
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        iterations = list(range(1, len(run_data['test_f1']) + 1))
        ax.plot(iterations, run_data['test_f1'], 'b-o', label='Test F1')
        ax.plot(iterations, run_data['validation_f1'], 'r--s', label='Validation F1')
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Run {run_idx+1} - F1 Score Progression")
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        ttk.Separator(scrollable_frame).grid(row=run_idx+2, column=0, sticky="ew", pady=10)

def ui():
    global root
    root = tk.Tk()
    root.title("Result charts view")
    root.geometry("1100x450")
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)
    
    upload_frame = ttk.LabelFrame(main_frame, text="Results Analysis")
    upload_frame.pack(pady=20, padx=20, fill="x")
    
    ttk.Button(upload_frame, text="Upload Results File", command=load_results).pack(pady=15, padx=50)
    
    root.mainloop()

if __name__ == '__main__':
    ui()