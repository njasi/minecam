# echo-server.py
import socket
import time
import json
import traceback
from xdo import Xdo
from autopilot.input import Mouse
from math import ceil


# import faulthandler
# faulthandler.enable()


xdo = Xdo()
MOUSE = Mouse.create()
PORT = 55555  # Port to listen on (non-privileged ports are > 1023)
file = open("src/pose/.host")
HOST = file.readline()  # Standard loopback interface address (localhost)
file.close()

print("You have 5 seconds to click on your minecraft window.")
# MOUSE.move(75,480)

TOP_BOUND = 100
LEFT_BOUND = 100

time.sleep(5)
win_id = xdo.select_window_with_click()
window_size = xdo.get_window_size(win_id)
midpt = (int(window_size[0] / 2), int(window_size[1] / 2))


def apply_action(action):
    if(action["action"] == "mouse_move"):
        x, y = MOUSE.position()

        print(x,y)
        try:
            x_new = x+action["x"]
            y_new = y+action["y"]
            if(x_new > window_size[0] or \
                x_new < LEFT_BOUND or \
                y_new > window_size[1] or \
                y_new < TOP_BOUND):
                xdo.move_mouse(midpt[0],midpt[1])
                x, y = MOUSE.position()

            x_new = x+action["x"]
            y_new = y+action["y"]
            MOUSE.move(x_new,y_new)
        except:
            print(x,y)
            print("\n\nMOUSE MOVE ERROR:\n")
            # print(traceback.format_exc())
            print()
            return
        pass
    elif(action["action"] == "keypress"):
        key = bytes(action["key"],encoding='utf8')
        print(key)
        xdo.enter_text_window(win_id, key)
    elif(action["action"] == "click"):
        # 4 => scroll up
        MOUSE.press(button=action["button"])
    elif(action["action"] == "click_release"):
        MOUSE.release(button=1)
        MOUSE.release(button=3)


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
                # print(action)
                if("action" in action):
                    apply_action(action)
            except:
                print("DATA ERROR:\t",data)
                # print(traceback.format_exc(),"\n")
                continue

            if not data:
                break