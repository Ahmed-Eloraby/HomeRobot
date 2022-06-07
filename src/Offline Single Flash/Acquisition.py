import UnicornPy
import numpy as np
import time
from ConfigUnicorn import *
def main():
    # Specifications for the data acquisition.
    # -------------------------------------------------------------------------------------
    TestsignaleEnabled = False;
    FrameLength = 1;  # number of readings
    # AcquisitionDurationInSeconds = 10;
    from os import path, mkdir, listdir
    if not path.isdir(f"{DATA_PATH}/{NAME}"):
        mkdir(f"{DATA_PATH}/{NAME}")
    if not path.isdir(f"{DATA_PATH}/{NAME}/{TRAIN_OR_TEST}"):
        mkdir(f"{DATA_PATH}/{NAME}/{TRAIN_OR_TEST}")

    count = 0

    # Iterate directory
    for path in listdir(f"{DATA_PATH}/{NAME}/{TRAIN_OR_TEST}"):
        # check if current path is a file
        count += 1

    DataFile = f"{DATA_PATH}/{NAME}/{TRAIN_OR_TEST}/BCI{('{:02d}'.format(int(count / 2) + 1))}.csv";
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        # Get available devices.
        # -------------------------------------------------------------------------------------

        deviceList = UnicornPy.GetAvailableDevices(True)
        print("connecting")

        deviceID = 0

        device = UnicornPy.Unicorn(deviceList[deviceID])
        print("Connected to '%s'." % deviceList[deviceID])

        # Create a file to store data.
        file = open(DataFile, "wb")

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
        receiveBufferBufferLength = numberOfAcquiredChannels * 4 * FrameLength
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
            # while not start_reading:
            #     # print(data_size)
            #     time.sleep(0.2)

            # global size
            # data_size = int(100000)
            #
            # targets = np.empty(data_size, dtype="S13")
            # rows = np.zeros(data_size, dtype=np.int8)
            # samples = np.zeros([data_size, 8], dtype=np.float32)

            i = int(0)
            while True:
                global current_trarget
                global current_col
                # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                # Convert receive buffer to numpy float array
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                print("data ", data)
                data=np.append(data,int(time.time_ns() // 1e6))

                data = np.reshape(data, (1, numberOfAcquiredChannels+1))

                # print(data.shape)
                # print(data)
                np.savetxt(file, data, delimiter=',', fmt='%.3f', newline='\n')
                # samples[i] = data[0:8]
                # targets[i] = current_trarget
                # rows[i] = current_col
                #
                # i += 1




            # Stop data acquisition.
            # -------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

            # from scipy.io import savemat
            # name = "ahmed"
            # trainortest = "train"
            # from os import mkdir, path
            # if not path.isdir(f"./experiment/{name}"):
            #     mkdir(f"./experiment/{name}")
            # savemat(f"./experiment/{name}/{name}{trainortest}.mat", {'Samples': samples, 'StimulusCode': rows,
            #                                                          'TargetInstruction': targets})


        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" % e)
        finally:
            # release receive allocated memory of receive buffer
            del receiveBuffer

            # close file
            file.close()

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