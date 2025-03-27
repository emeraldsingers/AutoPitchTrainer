import tkinter as tk
from tkinter import messagebox
import threading
import os
from .. import tasks
from ..utils import log_queue
from ..core.vocabulary import load_vocabulary # Нужно для start_conversion

class Handlers:
    def __init__(self, main_window):
        self.main_window = main_window

    def start_training(self):
        data_dir = self.main_window.train_data_dir_entry.get()
        output_dir = self.main_window.train_output_dir_entry.get()
        epochs = self.main_window.train_epochs_var.get()
        batch_size = self.main_window.train_batch_size_var.get()
        if not data_dir or not output_dir:
            messagebox.showerror("Error", "Please provide data and output directories.")
            return
        thread = threading.Thread(target=tasks.run_training, args=(data_dir, output_dir, epochs, batch_size, log_queue))
        thread.start()

    def start_fine_tuning(self):
        model_dir = self.main_window.ft_model_dir_entry.get()
        data_dir = self.main_window.ft_data_dir_entry.get()
        output_dir = self.main_window.ft_output_dir_entry.get()
        epochs = self.main_window.ft_epochs_var.get()
        batch_size = self.main_window.ft_batch_size_var.get()
        freeze_layers = self.main_window.ft_freeze_layers_var.get()
        if not model_dir or not data_dir or not output_dir:
            messagebox.showerror("Error", "Please provide all required fields.")
            return
        thread = threading.Thread(target=tasks.run_fine_tuning, args=(model_dir, data_dir, output_dir, epochs, batch_size, freeze_layers, log_queue))
        thread.start()

    def start_resume_training(self):
        model_dir = self.main_window.rt_model_dir_entry.get()
        data_dir = self.main_window.rt_data_dir_entry.get()
        output_dir = self.main_window.rt_output_dir_entry.get()
        epochs = self.main_window.rt_epochs_var.get()
        batch_size = self.main_window.rt_batch_size_var.get()
        if not model_dir or not data_dir or not output_dir:
            messagebox.showerror("Error", "Please provide all required fields.")
            return
        thread = threading.Thread(target=tasks.run_resume_training, args=(model_dir, data_dir, output_dir, epochs, batch_size, log_queue))
        thread.start()

    def start_processing(self):
        model_dir = self.main_window.proc_model_dir_entry.get()
        model_version = self.main_window.proc_model_version_var.get()
        input_ustx = self.main_window.proc_input_ustx_entry.get()
        output_dir = self.main_window.proc_output_dir_entry.get()
        if not model_dir or not input_ustx or not output_dir:
            messagebox.showerror("Error", "Please provide all required fields.")
            return
        model_file = os.path.join(model_dir, 'best_loss_autopitch_model.pth' if model_version == 'Best Loss' else 'last_autopitch_model.pth')
        vocab_file = os.path.join(model_dir, 'phoneme_vocab.yaml')
        if not os.path.exists(model_file) or not os.path.exists(vocab_file):
            messagebox.showerror("Error", "Model or vocabulary file not found.")
            return
        thread = threading.Thread(target=tasks.run_processing, args=(model_file, input_ustx, output_dir, vocab_file, log_queue))
        thread.start()

    def start_conversion(self):
        model_dir = self.main_window.conv_model_dir_entry.get()
        output_onnx_path = self.main_window.conv_output_entry.get()
        if not model_dir or not output_onnx_path:
            messagebox.showerror("Error", "Please provide model directory and output ONNX path.")
            return
        model_path = os.path.join(model_dir, 'last_autopitch_model.pth')
        vocab_file = os.path.join(model_dir, 'phoneme_vocab.yaml')
        if not os.path.exists(model_path) or not os.path.exists(vocab_file):
            messagebox.showerror("Error", "Model or vocabulary file not found.")
            return
        phoneme_to_idx, _ = load_vocabulary(vocab_file)
        thread = threading.Thread(target=tasks.run_conversion, args=(model_path, output_onnx_path, len(phoneme_to_idx), log_queue))
        thread.start()