
import requests

IP ="http://192.168.43.226/"

while True:
    x = input()
    requests.get(IP + str(x))

