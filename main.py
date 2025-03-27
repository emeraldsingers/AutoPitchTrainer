import tkinter as tk
from autopitch.gui.main_window import MainWindow
from autopitch.utils import log_queue, check_log_queue

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root, log_queue)
    check_log_queue(root, app.log_text, log_queue)
    root.mainloop()