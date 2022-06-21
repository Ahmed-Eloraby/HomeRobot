from threading import Thread
import time
import concurrent.futures
from threading import Lock
import multiprocessing
import asyncio

import numpy as np

from ConfigUnicorn import *


# all threads can access this global variable

from joblib import dump, load

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


scaler = load('Tools/scaler.joblib')
pca = load('Tools/pca.joblib')
model = load('Tools'
             '/model.joblib')

def headset(conn):
    import UnicornPy
    import numpy as np

    def main():
        out = []
        read = False



        # Specifications for the data acquisition.
        # -------------------------------------------------------------------------------------
        TestsignaleEnabled = False;
        FrameLength = 1;  # number of readings

        print("Unicorn Acquisition Example")
        print("---------------------------")
        print()

        try:

            deviceList = UnicornPy.GetAvailableDevices(True)
            print("connecting")

            deviceID = 0

            device = UnicornPy.Unicorn(deviceList[deviceID])
            print("Connected to '%s'." % deviceList[deviceID])

            numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()

            # Print acquisition configuration
            print("Acquisition Configuration:");
            print("Sampling Rate: %i Hz" % UnicornPy.SamplingRate);
            print("Frame Length: %i" % FrameLength);
            print("Number Of Acquired Channels: %i" % numberOfAcquiredChannels);

            # Allocate memory for the acquisition buffer.
            receiveBufferBufferLength = numberOfAcquiredChannels * 4 * FrameLength
            receiveBuffer = bytearray(receiveBufferBufferLength)

            try:
                # Start data acquisition.
                # -------------------------------------------------------------------------------------
                device.StartAcquisition(TestsignaleEnabled)
                print("Data acquisition started.")

                i = int(0)
                while True:

                    # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                    device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                    # Convert receive buffer to numpy float array
                    data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                    if read:
                        out.append(np.append(data, time.time_ns() / 1e6))
                    if conn.poll():
                        read = conn.recv()

                        if not read:
                            conn.send(out)
                            out = []
                        elif read == -1:
                            break



                # Stop data acquisition.
                # -------------------------------------------------------------------------------------
                device.StopAcquisition();
                print()
                print("Data acquisition stopped.");


            except UnicornPy.DeviceException as e:
                print(e)
            except Exception as e:
                print("An unknown error occured. %s" % e)
            finally:
                # release receive allocated memory of receive buffer
                del receiveBuffer

                # close file
                # file.close()

                # Close device.
                # -------------------------------------------------------------------------------------
                del device
                print("Disconnected from Unicorn")

        except Unicorn.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" % e)

    # execute main
    main()


# def dummy(conn):
#     def send_out():
#         t1 = Thread(target=conn.send, args=(out,))
#         t1.start()
#
#     import numpy as np
#     out = []
#     read = False
#     while True:
#         x = np.random.randint(0, 100000, 18, int)
#         if read:
#             out.append(np.append(x, time.time_ns() / 1e6))
#         if conn.poll():
#             read = conn.recv()
#             if not read:
#                 send_out()
#
#         time.sleep(0.004)

def predict_char(X,  prob_func=None):
    print(X)
    X = pca.transform(scaler.transform(X))
    if prob_func is None:
        prob_func = model.decision_function

    score = prob_func(X)
    print(score)
    row = np.argmax(score[3:])
    col = np.argmax(score[:3])
    predicted_char = INSTRUCTION_MATRIX[row][col]

    return predicted_char


def indeces_of_instruction(c):
    indeces = np.where(INSTRUCTION_MATRIX == c)
    row = indeces[0][0] + 4
    col = indeces[1][0] + 1
    return row, col


def prepare_for_model(signal, code):
    # if it has P300
    responses = np.zeros((NUMBER_OF_FLASHES, WINDOW * NUMBER_OF_CHANNELS), np.float32)
    # Looping over eache entry in our signal
    for i in range(1, signal.shape[0]):
        # Checking if flashing starts
        if code[i - 1] < 0.5 < code[i]:
            rowcol = int(code[i])

            for ch in CHANNELS:
                extracted_sample = signal[i:i + WINDOW, ch]
                responses[rowcol - 1][ch * WINDOW:ch * WINDOW + WINDOW] += extracted_sample

    responses = responses / NUMBER_OF_TRIALS
    return responses


def predict(UI_data, headset_data):
    UI_data = np.array(UI_data)
    headset_data = np.array(headset_data)
    my_data = headset_data
    initial_time = my_data[0, 17]
    ref_counter = my_data[0, 15]
    ref_time = initial_time
    time = my_data[:, 17]
    code = my_data[:, 15]

    bci_data = UI_data

    raw_data = np.copy(my_data[:, :8])

    from Preprocessing import preprocess_data
    preprocess_data(raw_data)

    data = raw_data

    j = int(0)
    stimuli = []

    for i in range(bci_data.shape[0]):
        if i < bci_data.shape[0] - 1:
            t = bci_data[i + 1, 2]
        else:
            t = int(time[-1] * 2)
        current_code = ref_counter + (t - ref_time) / 4 - 1
        while j < data.shape[0]:
            if current_code <= code[j]:
                k = j
                while time[k] == time[k - 1]:
                    k -= 1
                ref_counter = code[k]
                ref_time = time[k]
                break
            stimuli.append(bci_data[i, 0])
            j += 1

    stimuli = np.array(stimuli)
    responses = prepare_for_model(data, stimuli)

    return predict_char(responses)


def bci(conn):
    from itertools import cycle
    import requests

    import math
    import time
    import tkinter as tk
    import numpy as np

    IP = "http://192.168.43.226/"




    intensification_time = 94
    darken_time = 300
    epoch_break_time = 2000

    # number_of_trials = NUMBER_OF_TRIALS
    number_of_trials = 15

    intensified_font_size = 111
    darkened_font_size = 110

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

    MATRIX = arrange_data(INSTRUCTIONS)
    rows = MATRIX.shape[0]
    cols = MATRIX.shape[1]
    number_of_epochs = 1

    def get_row_col_encoding(number_of_rows, number_of_columns):
        row_col_encoding = np.arange(1, number_of_rows + number_of_columns + 1, dtype=np.int8)
        return row_col_encoding

    row_col_encoding = get_row_col_encoding(rows, cols)

    def get_row_col_order(row_col_encoding):
        row_col_order = np.array([], dtype=int)
        for i in range(number_of_trials):
            np.random.shuffle(row_col_encoding)
            row_col_order = np.append(row_col_order, row_col_encoding)

        return row_col_order

    def create_time_for_one_epoch():
        t = np.array([epoch_break_time], dtype=np.int16)
        one_trial = np.tile([intensification_time, darken_time], rows + cols)
        all_trials = np.tile(one_trial, number_of_trials)
        t = np.append(t, all_trials[:-1])
        return t

    def create_time_map():
        t = create_time_for_one_epoch()
        total = np.append(t, epoch_break_time)

        return total

    time_map = create_time_map()
    time_map[0] *= 5

    n = (rows + cols) * number_of_trials * 2 + 1
    # darkened_color = "#373737"
    darkened_color = "#1A1A1A"

    fonts = np.append(
        np.tile(np.array([darkened_font_size, intensified_font_size]),
                (rows + cols) * number_of_trials * number_of_epochs),
        darkened_font_size)

    def prepare_timeline():
        row_col_order = get_row_col_order(row_col_encoding)
        row_col_order = np.append(0, np.repeat(row_col_order, 2))

        color_order = np.full(n, darkened_color)
        intensified_colors = ["#FFFFFF", '#FF0000']
        total_intensified_colors = np.random.choice(intensified_colors,
                                                    (rows + cols) * number_of_trials * number_of_epochs,
                                                    replace=True)
        color_order[1::2] = total_intensified_colors

        time_line = np.empty(n, dtype=object)

        time_line[:] = (list(zip(cycle([0]), time_map, row_col_order, fonts, color_order)))

        return time_line

    def stop_fetching():
        conn.send(False)

    def start_fetching():
        conn.send(True)

    class Gui:
        label_color = "#777777"
        # background_color = "#232323"
        background_color = "#000000"
        counter_font = ("Helvetica", 50)

        def __init__(self, master, Matrix):
            self.counter_label = None
            self.counter = None
            self.main_frame = None
            self.master = master
            self.matrix = Matrix
            self.number_of_rows = Matrix.shape[0]
            self.number_of_cols = Matrix.shape[1]
            self.configure_master()
            self.screen_width = self.master.winfo_screenwidth()
            self.screen_height = self.master.winfo_screenheight()
            self.Index = int(0)

            self.switcher = {
                "Left": "🢀",
                "Right": "🢂",
                "Backward": "🢃",
                "Forward": "🢁",
                "Still": "⯃",
                "Backward-Right": "🢆",
                "Backward-Left": "🢇",
                "Forward-Left": "🢄",
                "Forward-Right": "🢅",
                "": " "
            }
            self.recorder = []
            self.reset_time()

        def clear_frame(self):
            for widgets in self.master.winfo_children():
                widgets.destroy()

        def reset_time(self):
            self.Index = int(0)
            self.time_line = prepare_timeline()
            self.n = len(self.time_line)

            self.clear_frame()
            self.recorder = []
            self.current_stimulus = 0
            self.start_counter()

        def configure_master(self):
            self.master.title("BCI")
            self.master.bind("<Escape>", lambda e: e.widget.quit())
            # self.master.bind("<space>", self.start_counter)
            self.master.config(cursor="none")
            self.master.configure(background=self.background_color)
            self.master.attributes('-fullscreen', True)
            self.master.resizable(False, False)

        def start_counter(self):
            self.master.unbind("<space>")
            self.counter = 3
            # start fetching
            start_fetching()
            self.counter_label = tk.Label(self.master, text=self.counter, foreground=self.label_color,
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
            screen_factor = 1
            self.main_frame = tk.Frame(self.master, background=self.background_color,
                                       padx=self.screen_width * (1 - screen_factor) / 6,
                                       pady=self.screen_height * (1 - screen_factor) / 8)
            self.main_frame.pack(expand=True, fill='both')
            self.screen_width *= screen_factor
            self.screen_height *= screen_factor
            self.Label_Matrix = np.empty([rows, cols], tk.Label)
            self.prepare_label_matrix()

            self.time = time.time_ns()
            self.onetime = time.time_ns()

            self.updateGlobals()

            self.main_frame.after(self.time_line[0][1], self.simulate)

        def updateGlobals(self):
            self.current_stimulus ^= int(self.time_line[self.Index][2])
            temp = [self.current_stimulus, self.time_line[self.Index][0], int(time.time_ns() // 1e6)]
            self.recorder.append(temp)

        def wait_for_unicorn(self):
            if not conn.poll():
                self.main_frame.after(3000, self.wait_for_unicorn)

            else:

                data = conn.recv()
                self.predict(data)
                self.main_frame.after(10000, self.reset_time)
            #
            # target = np.random.choice(INSTRUCTIONS, 1)[0]
            # print(target)
            # self.react(target)
            # self.main_frame.after(10000, self.reset_time)

        def predict(self, data):
            target = predict(self.recorder, data)
            print(f"predict: {target}")
            # target = np.random.choice(INSTRUCTIONS, 1)[0]

            self.react(target)

        def green_target(self, index):

            self.Label_Matrix[int(index / 3)][int(index % 3)].config(foreground="#00FF00")

        def hardware_reaction(self, index):
            requests.get(IP + str(index))
            print(f"send: {IP}{index}")
            # pass
        def react(self, target):
            results = np.where(INSTRUCTIONS == target)
            index = results[0][0]
            print(index)

            # Software
            self.green_target(index)
            # hardware:
            self.hardware_reaction(index)

        def simulate(self):
            self.Index += 1
            if self.Index >= self.n:
                stop_fetching()
                self.wait_for_unicorn()
            else:

                # finish = time.time_ns()
                # diff = finish - self.onetime
                # diff /= 1e6
                # print(diff, self.time_line[self.Index - 1][1])
                # print((diff - self.time_line[self.Index - 1][1]))
                # self.onetime = time.time_ns()

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
                self.updateGlobals()

                self.main_frame.after(self.time_line[self.Index][1] - 10, self.simulate)

        def prepare_label_matrix(self):
            for index, char in np.ndenumerate(MATRIX):
                row = index[0]
                col = index[1]
                self.Label_Matrix[row][col] = tk.Label(self.main_frame, text=self.switcher.get(MATRIX[row][col]),
                                                       foreground=self.time_line[0][4],
                                                       background=self.background_color,
                                                       font=("Helvetica", self.time_line[0][3]),
                                                       )
                self.Label_Matrix[row][col].grid(row=row, column=col, sticky='nsew',
                                                 pady=(self.screen_height / self.number_of_rows / 9,
                                                       self.screen_height / self.number_of_rows / 3))
                self.main_frame.grid_columnconfigure(col, weight=1)
                self.main_frame.grid_columnconfigure(row, weight=1)

    root = tk.Tk()
    my_gui = Gui(root, MATRIX)
    root.mainloop()
    conn.send(-1)


if __name__ == "__main__":
    # bci(None)
    conn_1, conn_2 = multiprocessing.Pipe()
    bci_process = multiprocessing.Process(target=bci, args=(conn_1,))
    # unicorn_process = multiprocessing.Process(target=headset, args=(conn_2,))
    bci_process.start()
    headset(conn_2)
    # unicorn_process.start()
    bci_process.join()
    # unicorn_process.join()
    # thread_bci = threading.Thread(target=bci, daemon=True)
    # thread_bci.start()
    # # thread_un = threading.Thread(target=test3,daemon=True)
    # # thread_un.start()
    # dummy()
    # lock = Lock()
    # headset(lock)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     executor.submit(bci, lock)
    #     executor.submit(headset, lock)
    #     # executor.submit(test1)
    #     # executor.submit(test2)
    #
    # # t1 = Thread(target=increase)
    # # t2 = Thread(target=increase)
    # #
    # # t1.start()
    # # t2.start()
    # #
    # # t1.join()
    # # t2.join()
    #
    # print('End value:', database_value)
    #
    # print('end main')
    # executor.shutdown()
