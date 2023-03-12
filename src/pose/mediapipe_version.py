
import cv2
import mediapipe as mp
import numpy as np

import time
import json
import traceback
import faulthandler
import socket
from math import ceil, sqrt

faulthandler.enable()

KNEE_ADJUST = 0.2 #relative frame size, less knee raising required
CLASP_DIST = 0.05 # distance hands need to be to be considered a clasp
MOUSE_SPEED = 100 # pixels / second
PORT = 55555  # The port used by the server

file = open("src/pose/.host")
HOST = file.readline()  # Standard loopback interface address (localhost)
file.close()


class StateHead:
    def __init__(self):
        self.x = 7
        self.y = 0

class State:

    def __init__(self):
        self.head = StateHead()
        self.handleft = 0
        self.handright = 0
        self.kneeleft = 0
        self.kneeright = 0
        self.click = False
        self.inventory = False
        self.frame = 0
        self.walking = False
        self.handdist = 100







while True:
    try:
        SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        SOCKET.connect((HOST, PORT))
        break
    except:
        print("Could not connect to interface. Waiting 5 seconds.")
        time.sleep(5)


def send_json(j):
    SOCKET.sendall(bytes(json.dumps(j),encoding="utf-8"))


def pose_tracking():

    # initialize Pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)



    # create capture object
    cap = cv2.VideoCapture()

    # for fps tracking
    prev_frame_time = 0
    new_frame_time = 0
    state = State()

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:
        state.frame +=1
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        # cv2.imshow("frame", frame)


        try:
            # convert the frame to RGB format
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Pose Estimation: RGB ERR")
            continue
        # process the RGB frame to get the result

        RGB.flags.writeable = False
        results_pose = pose.process(RGB)
        results_face = face_mesh.process(RGB)
        RGB.flags.writeable = True

        # face direction detection
        image = frame
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matriautopilotx
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get aprev_frame_time = 0

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                state.head.x = x
                state.head.y = y
                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < 5:
                    text = "Looking Down"
                elif x > 15:
                    text = "Looking Up"
                else:
                    text = "Looking Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "NO face detected, adjust your position", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            state.head.x = 7
            state.head.y = 0

        # Checking hand position

        state.handleft = 0
        state.handright = 0
        if results_pose.pose_landmarks:
            left_wrist = results_pose.pose_landmarks.landmark[15]
            right_wrist = results_pose.pose_landmarks.landmark[16]
            left_shoulder= results_pose.pose_landmarks.landmark[11]
            right_shoulder = results_pose.pose_landmarks.landmark[12]
            if(left_wrist.y < left_shoulder.y):
                state.handleft = 1
            if(right_wrist.y < right_shoulder.y):
                state.handright = 1

            state.handdist = sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2 )
            cv2.putText(frame, "L,R hand: {},{}".format(state.handleft,state.handright), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Hand Distance: {}".format(state.handleft,state.handright), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # same thing as before, but knees
        state.kneeleft = 0
        state.kneeright = 0
        if results_pose.pose_landmarks:
            left_hip = results_pose.pose_landmarks.landmark[23]
            right_hip = results_pose.pose_landmarks.landmark[24]
            left_knee = results_pose.pose_landmarks.landmark[25]
            right_knee = results_pose.pose_landmarks.landmark[26]
            if(left_knee.y - KNEE_ADJUST < left_hip.y):
                state.kneeleft = 1
            if(right_knee.y - KNEE_ADJUST < right_hip.y):
                state.kneeright = 1
            cv2.putText(frame, "L,R knee: {},{}".format(state.kneeleft,state.kneeright), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Calculating the fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        time_change = new_frame_time-prev_frame_time
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, "FPS: " + fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)


        # show the final output
        cv2.imshow('Output', frame)

        if(state.frame < 3):
            continue

        # Mouse movements and clicks
        shift = MOUSE_SPEED * time_change

        x_speed = 0
        y_speed = 0
        if state.head.y < -10:
            # Looking Right
            x_speed = -1 * (state.head.y + 10) / 3 * shift

        elif state.head.y > 10:
            # Looking Left
            x_speed = -1 * (state.head.y - 10) / 3 * shift

        if state.head.x < 5:
            # Looking Down

            y_speed = -1 * (state.head.x - 5) / 3 * shift
            pass
        elif state.head.x > 15:
            # Looking Up
            y_speed = -1 * (state.head.x - 15) / 3 * shift
            pass

        try:
            # print(state.head.x,state.head.y)
            # print(x_speed,y_speed)
            if(abs(x_speed) > 0 or abs(y_speed) > 0):
                print(state.head.x,state.head.y)
                move = {
                    "action":"mouse_move",
                    "x": min(ceil(x_speed),20),
                    "y": min(ceil(y_speed),20)
                }
                send_json(move)
        except:
            print("\n\nPOSE DETECTION INTERFACE ERROR:\n")
            print(traceback.format_exc())
            print()
            continue

        # inventory
        if not state.inventory and state.handright == 1 and state.handleft == 1:
            inventory = {
                "action":"keypress",
                "key": 'e'
            }
            send_json(inventory)
            state.inventory = True
            continue # avoid sending a click along with the inventory

        # inventory lock
        if state.handright == 0 or state.handleft == 0:
            state.inventory = False

        # release clicks
        if state.click and \
            state.handright == 0 and \
            state.handleft == 0 and \
            state.handdist < CLASP_DIST:
            state.click = False
            click = {
                "action":"click_release",
            }
            send_json(click)

        # hand clasp
        if not state.click and state.handdist < CLASP_DIST:
            state.click = True
            click = {
                "action":"click",
                "button":4
            }
            send_json(click)

        # right click
        if not state.click and state.handright:
            state.click = True
            click = {
                "action":"click",
                "button":3
            }
            send_json(click)

        # left click
        if not state.click and state.handleft:
            state.click = True
            click = {
                "action":"click",
                "button":1
            }
            send_json(click)

        # walking
        if not state.walking and (state.kneeleft or state.kneeright):
            knee = {
                "action":"knee_up"
            }
            send_json(knee)
            state.walking = True

        # stop walking
        if state.walking and (state.kneeleft == 0 and  state.kneeright == 0):
            knee = {
                "action":"knee_down"
            }
            send_json(knee)
            state.walking = False


        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


pose_tracking()


