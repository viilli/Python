from __future__ import division
import cv2
import numpy as np
import tkinter as tk


lines = []
with open("Parameters.txt") as file_in:
	for line in file_in:
		lines.append(int(line))
		print(int(line))
	file_in.close()

p1 = lines[6]
d1 = lines[7]

root = tk.Tk()

number1 = tk.StringVar()
number2 = tk.StringVar()


e = tk.Entry(root, textvariable=number1)
f = tk.Entry(root, textvariable=number2)
e.pack()
f.pack()
def myClick():

	global d1
	global p1
	d1= (int(str(number1.get())))
	p1= (int(str(number2.get())))

myButton = tk.Button(root, text="Valmis", command=myClick)
myButton.pack()

def nothing(*arg):
    pass

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

icol = (lines[0], lines[1], lines[2], lines[3], lines[4], lines[5])  # Blue



font = cv2.FONT_HERSHEY_SIMPLEX


cv2.namedWindow('colorTest')
cv2.namedWindow('colorTestCrop')

# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)


#cv2.createTrackbar('Distance', 'colorTestCrop', d1, 255, nothing)


#cv2.createTrackbar('Crop'   , 'colorTest', icol[6], 255, nothing)

#frame = cv2.imread('colour-circles-test.jpg')
i = 0
while True:
	ret, frame = cap.read(1)
	frame2 = frame
	#frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


	lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
	lowsat = cv2.getTrackbarPos('lowSat', 'colorTest')
	lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
	highHue = cv2.getTrackbarPos('highHue', 'colorTest')
	highSat = cv2.getTrackbarPos('highSat', 'colorTest')
	highVal = cv2.getTrackbarPos('highVal', 'colorTest')
	#d1 = cv2.getTrackbarPos('Distance', 'colorTestCrop')


	print(str(d1) + " Root")
	#d1 = (str(root))



	Crop = 255
	# Show the original image.
	#cv2.imshow('frame 1', frame)

	# Blur methods available, comment or uncomment to try different blur methods.
	frameBGR = cv2.GaussianBlur(frame, (9, 9), 100)


    # Show blurred image.
	#cv2.imshow('blurred 2', frameBGR)

    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
	hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    # HSV values to define a colour range.
	colorLow = np.array([lowHue, lowsat, lowVal])
	colorHigh = np.array([highHue, highSat, highVal])
	mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
	#cv2.imshow('mask-plain 3', mask)

	kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # Show morphological transformation mask
	#cv2.imshow('mask 4', mask)

    # Put mask over top of the original image.
	result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show final output image
	#cv2.imshow('colorTest 5', result)

	frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	colorLow = np.array([lowHue, lowsat, lowVal])
	colorHigh = np.array([highHue, highSat, highVal])
	mask = cv2.inRange(frameHSV, colorLow, colorHigh)

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	drawCont =  cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

	#cv2.imshow('drawContours 6', drawCont)

	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	biggest_contour = ""
	try:
		biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
	except:
		print("")
	drawCont2 = cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 3)
	#cv2.imshow('drawContours2 7', drawCont2)

	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	biggest_contour = ""
	x = 0
	y = 0
	w = 0
	h = 0

	CenterPosX = 0
	CenterPosY = 0

	DistanceZ = 0

	try:
		biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

		x, y, w, h = cv2.boundingRect(biggest_contour)


	except:
		print("Error25")

	rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

	if(Crop == 255):
		try:
			CropFrame = frame2[y:y+h, x: x+h]
			cv2.imshow("Cropped frame", CropFrame)
		except:
			print("Crop Error")
	#print(str(x) + " : " + str(y))


	try:
		CenterPosX = int((x + (x + w)) / 2)
		CenterPosY = int((y + (y + h)) / 2)
		#print(str(w) + " : " + str(h))
		rectangle = cv2.circle(rectangle, (x , y), 10, color=(0, 0, 255), thickness=-1)
	except:
		print("Cant make Circle")

	#print(str(CenterPosX) + " : " + str(CenterPosY))

	print(str(x) + " : " + str(y))

	# w = p
	#p1 = 384
	p3 = w

	#d1 = 14
	d3 = 0


	try:
		d3 = (p1 / p3) * d1

	except:
		print("Error 10")

	print(str(d3) + "Distance 3")

	rectangle = cv2.putText(rectangle, str(round(d3, 2)) + " CM", (x, y), font, 2, color=(0, 0, 0), thickness=2)
	rectangle = cv2.putText(rectangle, str(w) + " px", (x, y+ h + 40), font, 2, color=(0, 0, 0), thickness=2)

	cv2.imshow('rectangle', rectangle)

	k = cv2.waitKey(5) & 0xFF
	if  k == 107:
		root.mainloop()

	if k == 27:
		print("Exit")
		file = open("Parameters.txt", "w")
		file.write(str(lowHue) + "\n")
		file.write(str(lowsat) + "\n")
		file.write(str(lowVal) + "\n")
		file.write(str(highHue) + "\n")
		file.write(str(highSat) + "\n")
		file.write(str(highVal) + "\n")
		file.write(str(p1) + "\n")
		file.write(str(d1))

		file.close()
		#break

cv2.destroyAllWindows()
