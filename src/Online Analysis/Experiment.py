import multiprocessing
import time
from threading import Thread



import numpy
import numpy as np


def sender(conn, msgs):
    def publish_array():
        print("sending")
        t1 = Thread(target=conn.send,args=(x,))
        t1.start()

    """
    function to send messages to other end of pipe
    """
    x = []
    while True:
        time.sleep(0.2)
        x= np.random.randint(0,8,3)
        publish_array()
    # for i in range(100000):
    #     x.append(np.random.randint(10,20,18))
    # t = time.time_ns()/1e6
    # t1 = Thread(target=publish_array)
    # t1.start()
    # print(time.time_ns()/1e6-t)





def receiver(conn):
    """
    function to print the messages received from other
    end of pipe
    """
    while True:
        while not conn.poll():
            time.sleep(0.004)
        msg = conn.recv()
        print(msg)
        print(type(msg))

if __name__ == "__main__":
    # messages to be sent
    msgs = ["hello", "hey", "hru?", "END"]
    # print(np.random.choice(msgs,1))
    # creating a pipe
    parent_conn, child_conn = multiprocessing.Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=sender, args=(parent_conn, msgs))
    # p2 = multiprocessing.Process(target=receiver, args=(child_conn,))

    # running processes
    p1.start()
    # p2.start()
    receiver(child_conn)
    # wait until processes finish
    # p1.join()
    # p2.join()
    # receiver(child_conn)