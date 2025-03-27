import torch
import queue
import tkinter as tk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_queue = queue.Queue()

def update_log(log_widget, message):
    log_widget.insert(tk.END, message + '\n')
    log_widget.see(tk.END)

def check_log_queue(root, log_widget, q):
    while not q.empty():
        message = q.get()
        update_log(log_widget, message)
    root.after(100, check_log_queue, root, log_widget, q)

print("Using device: ", device)