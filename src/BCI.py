import math
import time
import tkinter
import tkinter as tk
import numpy as np

intensification_time = 100
darken_time = 300
epoch_break_time = 2000
number_of_trials = 15
number_of_epochs = 2
data = np.array(["Forward-Left", "Forward", "Forward-Right", "Left", "Still", "Right", "Backward-Left", "Backward",
                 "Backward-Right"])
intensified_font_size = 30
darkened_font_size = 15


def number_of_rows_cols(n):
    row = int(math.sqrt(n))
    col = row if row * row == n else row + 1
    if row * col < n:
        col += 1
    return row, col


def create_empty(n):
    return np.repeat(np.array([""]), n)


def arrange_data(data):
    row, col = number_of_rows_cols(len(data))
    newSize = row * col

    empty = create_empty(newSize - len(data))

    return np.append(data, empty).reshape(row, col)


MATRIX = arrange_data(data)
rows = MATRIX.shape[0]
cols = MATRIX.shape[1]

test_data = np.random.choice(data, number_of_epochs, True)
number_of_epochs = len(test_data)
test_data = np.repeat(test_data, number_of_trials * (rows + cols) * 2)
test_data = np.append(test_data, "")


def get_row_col_encoding(number_of_rows, number_of_columns):
    row_col_encoding = np.arange(1, number_of_rows + number_of_columns + 1, dtype=np.int8)
    return row_col_encoding


row_col_encoding = get_row_col_encoding(rows, cols)


def get_row_col_order(row_col_encoding):
    row_col_order = np.array([])
    for _ in range(number_of_epochs * number_of_trials):
        np.random.shuffle(row_col_encoding)
        row_col_order = np.append(row_col_order, row_col_encoding)
    return row_col_order


row_col_order = get_row_col_order(row_col_encoding)


def create_time_for_one_epoch():
    t = np.array([epoch_break_time], dtype=np.int16)
    one_trial = np.tile([intensification_time, darken_time], rows + cols)
    all_trials = np.tile(one_trial, number_of_trials)
    t = np.append(t, all_trials[:-1])
    return t


def create_time_map():
    t = create_time_for_one_epoch()
    total = np.append(np.tile(t, number_of_epochs), epoch_break_time)
    return total


time_map = create_time_map()
row_col_order = np.append(0, np.repeat(row_col_order, 2))

n = (rows + cols) * number_of_trials * number_of_epochs * 2 + 1
darkened_color = "#222222"
color_order = np.full(n, darkened_color)
intensified_colors = ["#FFFFFF", "#E8873C", "#FE1513", "#FEFF3B", "#E86C3D", "#FF004B"]
total_intensified_colors = np.random.choice(intensified_colors, (rows + cols) * number_of_trials * number_of_epochs,
                                            replace=True)
color_order[1::2] = total_intensified_colors

fonts = np.append(
    np.tile(np.array([darkened_font_size, intensified_font_size]), (rows + cols) * number_of_trials * number_of_epochs),
    darkened_font_size)
print(color_order)
print(fonts)
print(test_data)
print(row_col_order)

print(time_map.shape, row_col_order.shape, test_data.shape)


class Gui:
    label_color = "#CCCCCC"
    background_color = "#000000"
    counter_font = ("Helvetica", 50)

    def __init__(self, master, Matrix, time_line, s):
        self.s = s
        self.counter_label = None
        self.counter = None
        self.target_label = None
        self.main_frame = None
        self.master = master
        self.matrix = Matrix
        self.time_line = time_line
        self.number_of_rows = Matrix.shape[0]
        self.number_of_cols = Matrix.shape[1]
        self.configure_master()
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        self.Index = int(0)
        self.n = len(self.time_line)

    def configure_master(self):
        self.master.title("BCI")
        self.master.bind("<Escape>", lambda e: e.widget.quit())
        self.master.bind("<space>", self.start_counter)
        self.master.configure(background=self.background_color)
        self.master.attributes('-fullscreen', True)
        self.master.resizable(False, False)

    def start_counter(self, e):
        self.master.unbind("<space>")
        self.counter = 3
        self.counter_label = tk.Label(root, text=self.counter, foreground=self.label_color,
                                      background=self.background_color,
                                      font=self.counter_font)
        self.counter_label.pack(fill='both', expand=True)
        self.counter_label.after(1000, self.update_counter)

    def update_counter(self):

        self.counter = self.counter - 1
        if self.counter != 0:
            self.counter_label.config(text=self.counter)
            self.counter_label.after(1000, self.update_counter)
        else:
            self.counter_label.pack_forget()
            self.start_simulation()

    def start_simulation(self):
        self.target_label = tk.Label(root, text=self.time_line[0][0], foreground=self.label_color,
                                     background=self.background_color,
                                     font=self.counter_font, pady=(self.screen_height * 0.01))
        self.target_label.pack(fill="x")
        screen_factor = 0.9

        self.main_frame = tk.Frame(self.master, background=self.background_color,
                                   padx=self.screen_width * (1 - screen_factor) / 2,
                                   pady=self.screen_height * (1 - screen_factor) / 2)
        self.main_frame.pack(expand=True, fill='both')
        self.screen_width *= screen_factor
        self.screen_height *= screen_factor
        self.Label_Matrix = np.empty([rows, cols], tk.Label)
        self.prepare_label_matrix()
        self.time = time.time_ns()
        self.main_frame.after(self.time_line[0][1] - 10, self.simulate)

    def simulate(self):
        self.Index += 1
        if self.Index >= self.n:
            finish = time.time_ns()
            diff = finish - self.time
            diff /= 1e6
            print(diff, self.s)
            print((diff - self.s) / self.n)
            return
        line = int(self.time_line[self.Index][2] - 1)
        if line < self.number_of_cols:
            for label in self.Label_Matrix[:, line]:
                label.config(foreground=self.time_line[self.Index][4],
                             font=("Helvetica", self.time_line[self.Index][3]))
        else:
            line = line - self.number_of_cols
            for label in self.Label_Matrix[line]:
                label.config(foreground=self.time_line[self.Index][4],
                             font=("Helvetica", self.time_line[self.Index][3]))
        self.target_label.config(text=self.time_line[self.Index][0])
        self.main_frame.after(self.time_line[self.Index][1] - 10, self.simulate)

    def prepare_label_matrix(self):
        for index, char in np.ndenumerate(MATRIX):
            row = index[0]
            col = index[1]
            self.Label_Matrix[row][col] = tk.Label(self.main_frame, text=MATRIX[row][col],
                                                   foreground=self.time_line[0][4],
                                                   background=self.background_color,
                                                   font=("Helvetica", self.time_line[0][3]))
            self.Label_Matrix[row][col].grid(row=row, column=col, sticky='nsew',
                                             pady=(self.screen_height / self.number_of_rows / 4,
                                                   self.screen_height / self.number_of_rows / 2))
            self.main_frame.grid_columnconfigure(col, weight=1)
            self.main_frame.grid_columnconfigure(row, weight=1)


time_line = np.empty(n, dtype=object)

time_line[:] = (list(zip(test_data, time_map, row_col_order, fonts, color_order)))
print(
    time_line.shape)  # time_line = {"rows": row_col_order, "time": time_map, "chars": test_data, "color": color_order,"font": fonts}
print(time_line)
print()
sumtime = np.sum(time_map)

root = tk.Tk()
my_gui = Gui(root, MATRIX, time_line, sumtime)
root.mainloop()
