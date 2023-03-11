from autopilot.input import Mouse
from xdo import Xdo


xdo = Xdo()
MOUSE = Mouse.create()

while True:
    (x,y) = MOUSE.position()
    print(x,"\t",y)
    if(x < 408):
        print("MOUSE MOVE\n")
        # MOUSE.move(600,y)
        xdo.move_mouse(600,y)