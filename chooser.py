import tkinter as tk
from tkinter import ttk, filedialog
import json
import os

def save_settings(settings):
    with open(".settings.json", "w") as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists(".settings.json"):
        with open(".settings.json", "r") as f:
            return json.load(f)
    return {}

def select_dataset(dataset_var):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    # askopenfilename = window for file selection
    if file_path:
        dataset_var.set(file_path)

def submit(root, feature_vars, algo_var, dataset_var, chosen_features):
    chosen_features[:] = [feature for feature, var in feature_vars.items() if var.get()]
    save_settings({"features": chosen_features, "algorithm": algo_var.get(), "dataset": dataset_var.get()})
    root.quit()
    root.destroy()

def choose():
    root = tk.Tk()
    root.title("DSLR - Training settings")
    
    # ------------------------------------------------------------------------- #

    tk.Label(root, text="Choisissez les features:").pack(anchor="center", pady=10)
    features = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
        'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
        'Flying'
    ]
    chosen_features = []
    oldFeatures = load_settings()

    feature_vars = {feature: tk.BooleanVar() for feature in features}
    for feature, var in feature_vars.items():
        selected_features = oldFeatures.get("features", [])
        if feature in selected_features:
            var.set(1)
        tk.Checkbutton(root, text=feature, variable=var).pack(anchor="center")
    
    # ------------------------------------------------------------------------- #

    tk.Label(root, text="Choisissez l'algorithme:").pack(anchor="center", pady=10)
    algos = ["gd", "sgd", "mgd"]
    algo_value = oldFeatures.get("algorithm")
    algo_var = tk.StringVar(value=algo_value if algo_value in algos else "gd")
    
    for algo in algos:
        ttk.Radiobutton(root, text=algo, variable=algo_var, value=algo).pack(anchor="center")
    
    # ------------------------------------------------------------------------- #

    tk.Label(root, text="Choisissez le dataset:").pack(anchor="center", pady=10)
    dataset_var = tk.StringVar()
    if oldFeatures.get("dataset"):
        dataset_var.set(oldFeatures.get("dataset"))
    
    ttk.Button(root, text="Parcourir", command=lambda: select_dataset(dataset_var)).pack()
    tk.Label(root, textvariable=dataset_var).pack()

    # ------------------------------------------------------------------------- #

    accuracy_var = tk.BooleanVar()
    accuracy_var.set(True)
    ttk.Checkbutton(root, text="Calculer la précision", variable=accuracy_var).pack()

    # ------------------------------------------------------------------------- #
    

    ttk.Button(root, text="Valider", command=lambda: submit(root, feature_vars, algo_var, dataset_var, chosen_features)).pack(pady=50)

    # ------------------------------------------------------------------------- #

    root.mainloop()
    
    return chosen_features, algo_var.get(), dataset_var.get(), accuracy_var.get()
