import math
import time
import tkinter as tk
import numpy as np


def number_of_rows_cols(n):
    row = int(math.sqrt(n))
    col = row if row * row == n else row + 1
    return row, col


def create_empty(n):
    return np.repeat(np.array([""]), n)


def arrange_data(data):
    row, col = number_of_rows_cols(len(data))
    newSize = row * col

    empty = create_empty(newSize - len(data))

    return np.append(data, empty).reshape(row, col)


data = np.array(["A", "B", "C", "D", "E"])

MATRIX = arrange_data(data)

row_col_encoding = np.arange(1, MATRIX.shape[0] + MATRIX.shape[1] + 1)
print(row_col_encoding)


def start_counter():
    global counter
    counter = 3

    def update_counter():
        global counter

        counter = counter - 1
        if counter != 0:
            count.config(text=counter)
            count.after(1000, update_counter)
        else:
            count.pack_forget()

    count = tk.Label(root, text=counter, foreground = intensified_color, background=background_color,font=counter_font)
    count.pack(fill='both', expand=True)
    count.after(1000, update_counter)


#
# global counter
# counter = 3
#
global background_color
background_color = "#000000"
global intensified_color
intensified_color = "#FFFFFF"
global darkened_color
darkened_color = "#777777"
global counter_font
counter_font=("Helvetica", 120)
global font
font=("Helvetica", 50)

root = tk.Tk()
root.title("BCI")
root.bind("<Escape>", lambda e: e.widget.quit())
root.bind("<space>", start_counter)

root.configure(background=background_color)
root.state('zoomed')

# #getting screen width and height of display
# width= root.winfo_screenwidth()
# height= root.winfo_screenheight()
# #setting tkinter window size
# root.geometry("%dx%d" % (width, height))#getting screen width and height of display
# width= root.winfo_screenwidth()
# height= root.winfo_screenheight()
# #setting tkinter window size
# root.geometry("%dx%d" % (width, height))
root.resizable(False, False)
start_counter()
#
greeting = tk.Frame(root)
#
greeting.pack()
#
root.mainloop()
