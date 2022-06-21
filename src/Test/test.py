import requests
import sys
import json
import time
import math
import signal

ROBOTINOIP = "127.0.0.1"
# ROBOTINOIP = "192.168.1.223"
PARAMS = {'sid': 'example_circle'}
run = True


def signal_handler(sig, frame):
    global run
    print('You pressed Ctrl+C!')
    run = False


def set_vel(vel):
    OMNIDRIVE_URL = "http://" + ROBOTINOIP + "/data/omnidrive"
    r = requests.post(url=OMNIDRIVE_URL, params=PARAMS, json=vel)
    if r.status_code != requests.codes.ok:
        # print("Error: post to %s with params %s failed", OMNIDRIVE_URL, PARAMS)
        raise RuntimeError("Error: post to %s with params %s failed", OMNIDRIVE_URL, PARAMS)


def bumper():
    BUMPER_URL = "http://" + ROBOTINOIP + "/data/bumper"
    r = requests.get(url=BUMPER_URL, params=PARAMS)
    if r.status_code == requests.codes.ok:
        data = r.json()
        return data["value"]
    else:
        raise RuntimeError("Error: get from %s with params %s failed", BUMPER_URL, PARAMS)


def distances():
    DISTANCES_URL = "http://" + ROBOTINOIP + "/data/distancesensorarray"
    r = requests.get(url=DISTANCES_URL, params=PARAMS)
    if r.status_code == requests.codes.ok:
        data = r.json()
        return data
    else:
        raise RuntimeError("Error: get from %s with params %s failed", DISTANCES_URL, PARAMS)


# rotate tuple vec by deg degrees and return the rotated vector as a list
def rotate(vec, deg):
    rad = 2 * math.pi / 360 * deg;

    out = [0, 0];

    out[0] = (math.cos(rad) * vec[0] - math.sin(rad) * vec[1]);
    out[1] = (math.sin(rad) * vec[0] + math.cos(rad) * vec[1]);

    return out;


def main():
    signal.signal(signal.SIGINT, signal_handler)

    try:
        startVector = (0.2, 0.0);
        a = 0;
        msecsElapsed = 0;
        vec = [0, 0, 0];

        while False == bumper() and True == run:
            dir = rotate(startVector, a);
            a = 360.0 * msecsElapsed / 10000;

            vec[0] = dir[0];
            vec[1] = dir[1];

            set_vel(vec);

            time.sleep(0.05)
            msecsElapsed += 50;

        set_vel([0, 0, 0])

    except Exception as e:
        print(e)
        return 1

    return 0


if __name__ == "__main__":
    main()