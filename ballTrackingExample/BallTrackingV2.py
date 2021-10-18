from __future__ import division
import cv2
import numpy as np
import tkinter as tk
from numpy import ones,vstack
from numpy.linalg import lstsq


lines = []
with open("Parameters2.txt") as file_in:
	for line in file_in:
		lines.append(int(line))
		print(int(line))
	file_in.close()

root = tk.Tk()

number1 = tk.StringVar()
number2 = tk.StringVar()


e = tk.Entry(root, textvariable=number1, width=100)
f = tk.Entry(root, textvariable=number2, width=100)
e.pack()
f.pack()
def myClick():

	global d1
	global p1
	d1= (int(str(number1.get())))
	p1= (int(str(number2.get())))

myButton = tk.Button(root, text="Calibrate", command=myClick)
myButton.pack()

def nothing(*arg):
    pass

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

icol = (lines[0], lines[1], lines[2], lines[3], lines[4], lines[5])  # Blue



font = cv2.FONT_HERSHEY_SIMPLEX


cv2.namedWindow('colorTest')

# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

i = 0
while True:

	ret, frame = cap.read(1)
	ret2, frame2 = cap2.read(1)

	lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
	lowsat = cv2.getTrackbarPos('lowSat', 'colorTest')
	lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
	highHue = cv2.getTrackbarPos('highHue', 'colorTest')
	highSat = cv2.getTrackbarPos('highSat', 'colorTest')
	highVal = cv2.getTrackbarPos('highVal', 'colorTest')

	# Blur methods available, comment or uncomment to try different blur methods.
	frameBGR = cv2.GaussianBlur(frame, (9, 9), 100)
	frameBGR2 = cv2.GaussianBlur(frame2, (9, 9), 100)

    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
	hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(frameBGR2, cv2.COLOR_BGR2HSV)

    # HSV values to define a colour range.
	colorLow = np.array([lowHue, lowsat, lowVal])
	colorHigh = np.array([highHue, highSat, highVal])
	mask = cv2.inRange(hsv, colorLow, colorHigh)

	mask2 = cv2.inRange(hsv2, colorLow, colorHigh)

    # Show the first mask
	cv2.imshow('First mask', mask)
	cv2.imshow('First mask 2', mask2)

	kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

	mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernal)
	mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernal)

    # Show morphological transformation mask
	cv2.imshow('morphological transformation mask', mask)
	cv2.imshow('morphological transformation mask 2', mask2)

    # Put mask over top of the original image.
	result = cv2.bitwise_and(frame, frame, mask=mask)
	result2 = cv2.bitwise_and(frame2, frame2, mask=mask)

	frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	frameHSV2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

	colorLow = np.array([lowHue, lowsat, lowVal])
	colorHigh = np.array([highHue, highSat, highVal])

	mask = cv2.inRange(frameHSV, colorLow, colorHigh)
	mask2 = cv2.inRange(frameHSV2, colorLow, colorHigh)

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	drawCont =  cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
	drawCont2 =  cv2.drawContours(frame2, contours2, -1, (0, 255, 0), 3)

	cv2.imshow('drawContours', drawCont)
	cv2.imshow('drawContours 2', drawCont2)

	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	contour_sizes2 = [(cv2.contourArea(contour), contour) for contour in contours2]

	biggest_contour = ""
	biggest_contour2 = ""

	try:
		biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
	except:
		print("")
	try:
		biggest_contour2 = max(contour_sizes2, key=lambda x: x[0])[1]
	except:
		print("")

	drawCont_biggest = cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 3)
	drawCont_biggest2 = cv2.drawContours(frame2, biggest_contour2, -1, (0, 255, 0), 3)

	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	contour_sizes2 = [(cv2.contourArea(contour), contour) for contour in contours2]

	biggest_contour = ""
	x = 0
	y = 0
	w = 0
	h = 0

	x2 = 0
	y2 = 0
	w2 = 0
	h2 = 0

	CenterPosX = 0
	CenterPosY = 0

	CenterPosX2 = 0
	CenterPosY2 = 0

	DistanceZ = 0
	DistanceZ2 = 0

	try:
		biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
		x, y, w, h = cv2.boundingRect(biggest_contour)
	except:
		print("Error25")

	try:
		biggest_contour2 = max(contour_sizes2, key=lambda x: x[0])[1]

		x2, y2, w2, h2 = cv2.boundingRect(biggest_contour2)
	except:
		print("Error25")



	rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
	rectangle2 = cv2.rectangle(frame2, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 0), 2)


	try:
		CropFrame = frame[y:y+h, x: x+h]
		cv2.imshow("Cropped frame", CropFrame)
	except:
		print("Crop Error")
	try:
		CropFrame2 = frame2[y2:y2+h2, x2: x2+h2]
		cv2.imshow("Cropped frame 2", CropFrame2)
	except:
		print("Crop Error")



	try:
		CenterPosX = int((x + (x + w)) / 2)
		CenterPosY = int((y + (y + h)) / 2)
		rectangle = cv2.circle(rectangle, (CenterPosX , CenterPosY), 10, color=(0, 0, 255), thickness=-1)
	except:
		print("Cant make Circle")

	try:
		CenterPosX2 = int((x2 + (x2 + w2)) / 2)
		CenterPosY2 = int((y2 + (y2 + h2)) / 2)
		rectangle2 = cv2.circle(rectangle2, (CenterPosX2 , CenterPosY2), 10, color=(0, 0, 255), thickness=-1)
	except:
		print("Cant make Circle")

	#print(str(x) + " : " + str(y) + "    VIDEO 1")
	#print(str(x2) + " : " + str(y2) + "    VIDEO 2")

	rectangle = cv2.putText(rectangle, str(w) + " px", (x, y+ h + 40), font, 2, color=(0, 0, 0), thickness=2)
	rectangle2 = cv2.putText(rectangle2, str(w2) + " px", (x2, y2 + h2 + 40), font, 2, color=(0, 0, 0), thickness=2)


	rectangle2 = cv2.putText(rectangle2, str(x2) + "  :  " + str(y2) + " px", (x2, y2), font, 0.5, color=(0, 0, 0), thickness=2)
	rectangle = cv2.putText(rectangle, str(x) + "  :  " + str(y) + " px", (x, y), font, 0.5, color=(0, 0, 0), thickness=2)

	cv2.imshow('rectangle', rectangle)
	cv2.imshow('rectangle 2', rectangle2)

	h_cam1, w_cam1, c_cam1 = frame.shape
	h_cam2, w_cam2, c_cam2 = frame2.shape


	cam1PosXZ = (0, 0)  # cm
	cam2PosXZ = (0, 13)  # cm

	w_cam1C = w_cam1 / 2
	w_cam2C = w_cam2 / 2

	cam1Points = [[0.0, 0.0], [0.0, 0.22]]

	#print(str(x) + " X")
	if int(x) < 320:
		w_cam1CR = w_cam1C - x
		#print("< 320")
	elif int(x) > 320:
		w_cam1CR = w_cam1C + x
		#print("> 320")

	wInCm = w_cam1CR * 0.0264583333

	cam1Points[1][0] = cam1Points[1][0] + wInCm

	#print(str(cam1PosXZ) + "Cam pos 1 : " + str(w_cam1CR))

	cam1FocalLenght = 22 	#	mm
	cam2FocalLenght = 2.32	#	mm

	cam1Rot = (0, 0, 0)
	cam2Rot = (0, 0, 0)



	x_coordsCam1, y_coordsCam1 = zip(*cam1Points)

	A = vstack([x_coordsCam1,ones(len(x_coordsCam1))]).T
	m, c = lstsq(A, y_coordsCam1)[0]
	print("Line Solution is y = {m}x + {c}".format(m=m,c=c))



	#---------------- Yhtälö laskut ----------------------------#

	y = 0
	x = 0

	x = c / m

	y = c

	print(str(y) + " + " str(x))

	startPoint = (0, round(y), 3)
	endPoint = (round(x), 3), 0)

	try:
		rectangle = cv2.line(rectangle, startPoint, endPoint, (0, 0, 0), thickness=2)
	except:
		print("line error")


	k = cv2.waitKey(5) & 0xFF
	if  k == 107:
		root.mainloop()

	if k == 27:
		print("Exit")
		file = open("Parameters2.txt", "w")
		file.write(str(lowHue) + "\n")
		file.write(str(lowsat) + "\n")
		file.write(str(lowVal) + "\n")
		file.write(str(highHue) + "\n")
		file.write(str(highSat) + "\n")
		file.write(str(highVal) + "\n")
		file.write(str(p1) + "\n")
		file.write(str(d1))

		file.close()
		break

cv2.destroyAllWindows()
