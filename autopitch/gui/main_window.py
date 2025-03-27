import tkinter as tk
from tkinter import ttk, filedialog
from .event_handlers import Handlers

class MainWindow:
    def __init__(self, root, log_queue):
        self.root = root
        self.log_queue = log_queue
        self.root.title("AutoPitch Trainer v2.0 by asoqwer")
        self.root.geometry("600x600")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.train_frame = ttk.Frame(self.notebook)
        self.fine_tune_frame = ttk.Frame(self.notebook)
        self.resume_frame = ttk.Frame(self.notebook)
        self.process_frame = ttk.Frame(self.notebook)
        self.convert_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.train_frame, text='Train')
        self.notebook.add(self.fine_tune_frame, text='Fine-Tune')
        self.notebook.add(self.resume_frame, text='Resume Training')
        self.notebook.add(self.process_frame, text='Process')
        self.notebook.add(self.convert_frame, text='Convert to ONNX')

        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill='both', expand=True)
        scrollbar = tk.Scrollbar(self.log_text)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')

        self.handlers = Handlers(self)

        self._create_train_widgets()
        self._create_fine_tune_widgets()
        self._create_resume_widgets()
        self._create_process_widgets()
        self._create_convert_widgets()


    def _create_dir_browse_row(self, parent, label_text, row_index, command):
        tk.Label(parent, text=label_text).grid(row=row_index, column=0, sticky='e', padx=5, pady=5)
        entry = tk.Entry(parent, width=50)
        entry.grid(row=row_index, column=1, padx=5, pady=5)
        button = tk.Button(parent, text="Browse", command=lambda e=entry: command(e))
        button.grid(row=row_index, column=2, padx=5, pady=5)
        return entry

    def _browse_directory(self, entry_widget):
        dir_path = filedialog.askdirectory()
        if dir_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, dir_path)

    def _browse_file_or_dir(self, entry_widget):
         path = filedialog.askopenfilename(filetypes=[("USTX files", "*.ustx")]) or filedialog.askdirectory()
         if path:
             entry_widget.delete(0, tk.END)
             entry_widget.insert(0, path)

    def _browse_save_as(self, entry_widget, defaultextension, filetypes):
        file_path = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def _create_train_widgets(self):
        self.train_data_dir_entry = self._create_dir_browse_row(self.train_frame, "Data Directory:", 0, self._browse_directory)
        self.train_output_dir_entry = self._create_dir_browse_row(self.train_frame, "Output Directory:", 1, self._browse_directory)

        tk.Label(self.train_frame, text="Epochs:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.train_epochs_var = tk.IntVar(value=125)
        tk.Spinbox(self.train_frame, from_=1, to=10000, textvariable=self.train_epochs_var).grid(row=2, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.train_frame, text="Batch Size:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.train_batch_size_var = tk.IntVar(value=16)
        tk.Spinbox(self.train_frame, from_=1, to=128, textvariable=self.train_batch_size_var).grid(row=3, column=1, padx=5, pady=5, sticky='w')

        tk.Button(self.train_frame, text="Start Training", command=self.handlers.start_training).grid(row=4, column=1, pady=10)

    def _create_fine_tune_widgets(self):
        self.ft_model_dir_entry = self._create_dir_browse_row(self.fine_tune_frame, "Model Directory:", 0, self._browse_directory)
        self.ft_data_dir_entry = self._create_dir_browse_row(self.fine_tune_frame, "Data Directory:", 1, self._browse_directory)
        self.ft_output_dir_entry = self._create_dir_browse_row(self.fine_tune_frame, "Output Directory:", 2, self._browse_directory)

        tk.Label(self.fine_tune_frame, text="Epochs:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.ft_epochs_var = tk.IntVar(value=100)
        tk.Spinbox(self.fine_tune_frame, from_=1, to=10000, textvariable=self.ft_epochs_var).grid(row=3, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.fine_tune_frame, text="Batch Size:").grid(row=4, column=0, sticky='e', padx=5, pady=5)
        self.ft_batch_size_var = tk.IntVar(value=16)
        tk.Spinbox(self.fine_tune_frame, from_=1, to=128, textvariable=self.ft_batch_size_var).grid(row=4, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.fine_tune_frame, text="Freeze Layers:").grid(row=5, column=0, sticky='e', padx=5, pady=5)
        self.ft_freeze_layers_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.fine_tune_frame, variable=self.ft_freeze_layers_var).grid(row=5, column=1, padx=5, pady=5, sticky='w')

        tk.Button(self.fine_tune_frame, text="Start Fine-Tuning", command=self.handlers.start_fine_tuning).grid(row=6, column=1, pady=10)


    def _create_resume_widgets(self):
        self.rt_model_dir_entry = self._create_dir_browse_row(self.resume_frame, "Model Directory:", 0, self._browse_directory)
        self.rt_data_dir_entry = self._create_dir_browse_row(self.resume_frame, "Data Directory:", 1, self._browse_directory)
        self.rt_output_dir_entry = self._create_dir_browse_row(self.resume_frame, "Output Directory:", 2, self._browse_directory)

        tk.Label(self.resume_frame, text="Epochs:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.rt_epochs_var = tk.IntVar(value=100)
        tk.Spinbox(self.resume_frame, from_=1, to=10000, textvariable=self.rt_epochs_var).grid(row=3, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.resume_frame, text="Batch Size:").grid(row=4, column=0, sticky='e', padx=5, pady=5)
        self.rt_batch_size_var = tk.IntVar(value=16)
        tk.Spinbox(self.resume_frame, from_=1, to=128, textvariable=self.rt_batch_size_var).grid(row=4, column=1, padx=5, pady=5, sticky='w')

        tk.Button(self.resume_frame, text="Start Resume Training", command=self.handlers.start_resume_training).grid(row=5, column=1, pady=10)


    def _create_process_widgets(self):
        self.proc_model_dir_entry = self._create_dir_browse_row(self.process_frame, "Model Directory:", 0, self._browse_directory)

        tk.Label(self.process_frame, text="Model Version:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.proc_model_version_var = tk.StringVar(value='Best Loss') 
        ttk.Combobox(self.process_frame, textvariable=self.proc_model_version_var, values=['Best Loss'], state="readonly").grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.process_frame, text="Input USTX (file or dir):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.proc_input_ustx_entry = tk.Entry(self.process_frame, width=50)
        self.proc_input_ustx_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.process_frame, text="Browse", command=lambda e=self.proc_input_ustx_entry: self._browse_file_or_dir(e)).grid(row=2, column=2, padx=5, pady=5)

        self.proc_output_dir_entry = self._create_dir_browse_row(self.process_frame, "Output Directory:", 3, self._browse_directory)

        tk.Button(self.process_frame, text="Process", command=self.handlers.start_processing).grid(row=4, column=1, pady=10)

    def _create_convert_widgets(self):
        self.conv_model_dir_entry = self._create_dir_browse_row(self.convert_frame, "Model Directory:", 0, self._browse_directory)

        tk.Label(self.convert_frame, text="Output ONNX Path:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.conv_output_entry = tk.Entry(self.convert_frame, width=50)
        self.conv_output_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.convert_frame, text="Browse", command=lambda e=self.conv_output_entry: self._browse_save_as(e, ".onnx", [("ONNX files", "*.onnx")])).grid(row=1, column=2, padx=5, pady=5)

        tk.Button(self.convert_frame, text="Convert to ONNX", command=self.handlers.start_conversion).grid(row=2, column=1, pady=10)