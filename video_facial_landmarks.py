# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import FileVideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

from imutils.video import FPS
 
 # Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (640,360))

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

text_file = open("output.txt", "w")
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] loading the video...")
vs = FileVideoStream("1.mp4").start()
time.sleep(2.0)

fps = FPS().start()
start = time.time()
curr_frame = 0

# loop over the frames from the video stream
while vs.more():
#while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	curr_time = time.time() - start
	curr_frame = curr_frame + 1

	text_file.write(f"{curr_time}; {curr_frame}; {fps};")

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			text_file.write(f"{x};{y}; ")
			
	  
	cv2.putText(frame, f"{toFixed(curr_time, 1)}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1  )
	cv2.putText(frame, f"{curr_frame}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1  )
	
	# show the frame
	cv2.imshow("Frame", frame)
	cv2.imwrite(f"out{curr_frame}" + ".jpg", frame)

	fps.update()
	key = cv2.waitKey(1) & 0xFF
	text_file.write("\n")
	out.write(frame)



	if cv2.waitKey(1) == 27:
                break
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

text_file.close()
fps.stop()


# do a bit of cleanup

out.release()

cv2.destroyAllWindows()
vs.stop()
