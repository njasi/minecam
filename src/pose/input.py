# echo-server.py
import socket
import time
import json
from xdo import Xdo
from autopilot.input import Mouse
from math import ceil


file = open("src/pose/host.txt")

xdo = Xdo()
MOUSE = Mouse.create()
HOST = file.readline()  # Standard loopback interface address (localhost)
PORT = 55555  # Port to listen on (non-privileged ports are > 1023)
file.close()

print(HOST)

time.sleep(5)
win_id = xdo.select_window_with_click()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            action = {}
            try:
                action = json.loads(data)
                print(action)
            except:
                print("DATA ERROR:\t",data)
                continue

            if("action" not in action):
                continue

            if(action["action"] == "mouse_move"):
                x, y = MOUSE.position()
                MOUSE.move(x+action["x"],y+action["y"])
                pass
            elif(action["action"] == "keypress"):
                key = bytes(action["key"])
                print(key)
                xdo.enter_text_window(win_id, key)
                pass
            elif(action["action"] == "click"):
                # 4 => scroll up
                MOUSE.press(button=action["button"])
            elif(action["action"] == "click_release"):
                MOUSE.release(button=1)
                MOUSE.release(button=3)
            if not data:
                break
