
import cv2
import mediapipe as mp
import numpy as np
import time

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



# define a video capture object
vid = cv2.VideoCapture(0)

while True:
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
    results = pose.process(RGB)
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

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
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

            # print(y)

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < 10:
                text = "Looking Down"
            elif x > 20:
                text = "Looking Up"
            else:
                text = "Looking Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # print(results.pose_landmarks)


    mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

  
    # Calculating the fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, "FPS: " + fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 2)




    # show the final output
    cv2.imshow('Output', frame)



    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        breakframe

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

