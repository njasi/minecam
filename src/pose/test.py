import cv2
from headpose.detect import PoseEstimator

est = PoseEstimator()  #load the model
# take an image using the webcam (alternatively, you could load an image)

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # frame = cv2.resize(frame, (224, 224))


    cv2.waitKey(0)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", image)
    est.detect_landmarks(image, plot=True)
    # Display the resulting frame
    print("a")

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


exit()

cam = cv2.VideoCapture()
for i in range(cv2.CAP_PROP_FRAME_COUNT):
    cam.grab()
ret, image = cam.retrieve()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("showing image")
cv2.imshow("test", image)
cv2.waitKey(0)

cam.release()

# est.detect_landmarks(image, plot=True)  # plot the result of landmark detection
# roll, pitch, yawn = est.pose_from_image(image)  # estimate the head pose