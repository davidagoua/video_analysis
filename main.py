import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
img_ph = cap.read()
tracked_img = img_ph[100:300, 100:300]
tracked_img_grey = cv.cvtColor(tracked_img, cv.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if ret :
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        template = cv.matchTemplate(frame_grey, tracked_img_grey,cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(template)
        top_left = max_loc
        bottom_right = (top_left[0]+200, top_left[1]+200)
        res = cv.rectangle(frame, top_left, bottom_right, (255,255,255), 3)
        cv.imshow("Frame", frame)
        cv.imshow("Image tracked",tracked_img)

        if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()