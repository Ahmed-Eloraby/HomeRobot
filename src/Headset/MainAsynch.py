import time
from threading
import asyncio

# all threads can access this global variable
start_reading = True
database_value = 0
current_trarget = ""
current_col = 0

size = 0


def increase(lock):
    global start_reading
    while not start_reading:
        time.sleep(0.2)
        pass
    while True:
        global current_col
        print("Current Flash: ", current_col)
        global current_trarget
        print("Current Target: ", current_trarget)
        # # global database_value  # needed to modify the global value
        # print("database_value: ", database_value)
        # # get a local copy (simulate data retrieving)
        # local_copy = database_value
        #
        # # simulate some modifying operation
        # local_copy += 1
        #
        #
        # # write the calculated new value into the global variable
        # database_value = local_copy
        time.sleep(0.004)


def headset(lock):
    import UnicornPy
    import numpy as np

    def main():
        # Specifications for the data acquisition.
        # -------------------------------------------------------------------------------------
        TestsignaleEnabled = False;
        FrameLength = 1;  # number of readings
        # AcquisitionDurationInSeconds = 10;
        # DataFile = "data.csv";

        print("Unicorn Acquisition Example")
        print("---------------------------")
        print()

        try:
            # Get available devices.
            # -------------------------------------------------------------------------------------

            # Get available device serials.
            deviceList = UnicornPy.GetAvailableDevices(True)
            print("connecting")
            #
            # if len(deviceList) <= 0 or deviceList is None:
            #     raise Exception("No device available.Please pair with a Unicorn first.")
            #
            # # Print available device serials.
            # print("Available devices:")
            # i = 0
            # for device in deviceList:
            #     print("#%i %s" % (i, device))
            #     i += 1

            # Request device selection.
            # print()
            # deviceID = int(input("Select device by ID #"))
            deviceID = 0
            # if deviceID < 0 or deviceID > len(deviceList):
            #     raise IndexError('The selected device ID is not valid.')

            # Open selected device.
            # -------------------------------------------------------------------------------------
            # print()
            # print("Trying to connect to '%s'." % deviceList[deviceID])
            device = UnicornPy.Unicorn(deviceList[deviceID])
            print("Connected to '%s'." % deviceList[deviceID])
            # print()

            # Create a file to store data.
            # file = open(DataFile, "wb")

            # Initialize acquisition members.
            # -------------------------------------------------------------------------------------
            numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
            configuration = device.GetConfiguration()

            # Print acquisition configuration
            print("Acquisition Configuration:");
            print("Sampling Rate: %i Hz" % UnicornPy.SamplingRate);
            print("Frame Length: %i" % FrameLength);
            print("Number Of Acquired Channels: %i" % numberOfAcquiredChannels);
            # print("Data Acquisition Length: %i s" % AcquisitionDurationInSeconds);
            print();


            # Allocate memory for the acquisition buffer.
            receiveBufferBufferLength = 17 * 4 * 10
            receiveBuffer = bytearray(receiveBufferBufferLength)

            try:
                # Start data acquisition.
                # -------------------------------------------------------------------------------------
                device.StartAcquisition(TestsignaleEnabled)
                print("Data acquisition started.")

                # Calculate number of get data calls.
                # numberOfGetDataCalls = int(AcquisitionDurationInSeconds * UnicornPy.SamplingRate / FrameLength);

                # Acquisition loop.
                # -------------------------------------------------------------------------------------
                while not start_reading:
                    # print(data_size)
                    time.sleep(0.2)

                global size
                data_size = int(100000)

                targets = np.empty(data_size, dtype="S13")
                rows = np.zeros(data_size, dtype=np.int8)
                samples = np.zeros([data_size, 8], dtype=np.float32)

                i = int(0)
                while start_reading:
                    global current_trarget
                    global current_col
                    # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                    device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                    # Convert receive buffer to numpy float array
                    data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                    # print("data ", data)
                    # data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
                    # print(data.shape)
                    # print(data)
                    # np.savetxt(file, data, delimiter=',', fmt='%.3f', newline='\n')
                    samples[i] = data[0:8]
                    targets[i] = current_trarget
                    rows[i] = current_col

                    i += 1




                # Stop data acquisition.
                # -------------------------------------------------------------------------------------
                device.StopAcquisition();
                print()
                print("Data acquisition stopped.");

                from scipy.io import savemat
                name = "ahmed"
                trainortest = "train"
                from os import mkdir, path
                if not path.isdir(f"./experiment/{name}"):
                    mkdir(f"./experiment/{name}")
                savemat(f"./experiment/{name}/{name}{trainortest}.mat", {'Samples': samples, 'StimulusCode': rows,
                                                                         'TargetInstruction': targets})


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


async def bci():
    import math
    import time
    import tkinter as tk
    import numpy as np

    intensification_time = 94
    darken_time = 300
    epoch_break_time = 2000
    number_of_trials = 15
    number_of_epochs = 50
    data = np.array(["Forward-Left", "Forward", "Forward-Right", "Left", "Still", "Right", "Backward-Left", "Backward",
                     "Backward-Right"])
    intensified_font_size = 27
    darkened_font_size = 25

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
    darkened_color = "#373737"
    color_order = np.full(n, darkened_color)
    intensified_colors = ["#B464EB", "#FEFF3B", "#00FFF6"]
    total_intensified_colors = np.random.choice(intensified_colors, (rows + cols) * number_of_trials * number_of_epochs,
                                                replace=True)
    color_order[1::2] = total_intensified_colors

    fonts = np.append(
        np.tile(np.array([darkened_font_size, intensified_font_size]),
                (rows + cols) * number_of_trials * number_of_epochs),
        darkened_font_size)
    # print(color_order)
    # print(fonts)
    # print(test_data)
    # print(row_col_order)
    #
    # print(time_map.shape, row_col_order.shape, test_data.shape)

    class Gui:
        label_color = "#CCCCCC"
        background_color = "#232323"
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

        async def start_counter(self, e):
            self.master.unbind("<space>")
            self.counter = 3
            self.counter_label = tk.Label(root, text=self.counter, foreground=self.label_color,
                                          background=self.background_color,
                                          font=self.counter_font)
            self.counter_label.pack(fill='both', expand=True)
            await asyncio.sleep(1)
            self.counter_label.after( 0,self.update_counter)

        async def update_counter(self):
            self.counter = self.counter - 1
            if self.counter != 0:
                self.counter_label.config(text=self.counter)
                await asyncio.sleep(1)
                self.counter_label.after(0, self.update_counter)
            else:
                global start_reading
                start_reading = True
                self.counter_label.pack_forget()
                await self.start_simulation()

        async def start_simulation(self):
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
            self.onetime = time.time_ns()

            self.updateGlobals()
            await asyncio.sleep(0.1)
            self.main_frame.after(self.time_line[0][1], self.simulate)

        def updateGlobals(self):
            global current_col
            current_col ^= int(self.time_line[self.Index][2])
            global current_trarget
            current_trarget = self.time_line[self.Index][0]

        async def simulate(self):
            self.Index += 1
            if self.Index >= self.n:
                finish = time.time_ns()
                diff = finish - self.time
                diff /= 1e6
                # print(diff, self.s)
                # print((diff - self.s) / self.n)
                global start_reading
                start_reading = False
                return
            finish = time.time_ns()
            diff = finish - self.onetime
            diff /= 1e6
            # print(diff, self.time_line[self.Index - 1][1])
            # print((diff - self.time_line[self.Index - 1][1]) / self.n)
            self.onetime = time.time_ns()

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
            self.updateGlobals()
            await asyncio.sleep(0.1)
            self.main_frame.after(self.time_line[self.Index][1] - 10, self.simulate)

        def prepare_label_matrix(self):
            for index, char in np.ndenumerate(MATRIX):
                row = index[0]
                col = index[1]
                self.Label_Matrix[row][col] = tk.Label(self.main_frame, text=MATRIX[row][col],
                                                       foreground=self.time_line[0][4],
                                                       background=self.background_color,
                                                       font=("Helvetica", self.time_line[0][3]),
                                                       )
                self.Label_Matrix[row][col].grid(row=row, column=col, sticky='nsew',
                                                 pady=(self.screen_height / self.number_of_rows / 3,
                                                       self.screen_height / self.number_of_rows / 2))
                self.main_frame.grid_columnconfigure(col, weight=1)
                self.main_frame.grid_columnconfigure(row, weight=1)

    time_line = np.empty(n, dtype=object)

    time_line[:] = (list(zip(test_data, time_map, row_col_order, fonts, color_order)))
    # print(
    #     time_line.shape)  # time_line = {"rows": row_col_order, "time": time_map, "chars": test_data, "color": color_order,"font": fonts}
    # print(time_line)
    # print()
    sumtime = np.sum(time_map)
    global size
    size = sumtime
    root = tk.Tk()
    my_gui =await Gui(root, MATRIX, time_line, sumtime)
    root.mainloop()

async def test1():
    while True:
        print("Balabizo")
        await asyncio.sleep(.3)



def main(loop):
    try:
        asyncio.ensure_future(firstWorker())
        asyncio.ensure_future(secondWorker())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()

if __name__ == "__main__":
    # print('Start value: ', database_value)
    loop = asyncio.get_event_loop()
    loop.run_until_complete()
    main(loop)
    asyncio.run(main())

    # with concurrent.futures.ThreadPoolExecutor() as executor:
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


