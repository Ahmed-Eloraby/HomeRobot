import numpy as np

NAME = "Omar Osama".lower()

INSTRUCTION_MATRIX =np.array([["Forward-Left", "Forward", "Forward-Right"],[ "Left", "Still", "Right"],
                                      ["Backward-Left", "Backward","Backward-Right"]])

NSTRUCTIONS = np.array(["Forward-Left", "Forward", "Forward-Right", "Left", "Still", "Right", "Backward-Left", "Backward",
                 "Backward-Right"])

NUMBER_OF_EPOCHS =50

CHANNEL = 10

CHANNELS = np.arange(8)

NUMBER_OF_CHANNELS = 8

DATA_PATH = "./experiments"

SAMPLING_FREQUENCY = 250

WINDOW = 250

LOWCUT_FREQ = 0.1
HIGHCUT_FREQ = 20

NUMBER_OF_FLASHES = 6
NUMBER_OF_TRIALS = 15
#notch
NOTCH_FREQ = 50
QUALITY_FACTOR = 20

TRAIN_OR_TEST = "Train"
