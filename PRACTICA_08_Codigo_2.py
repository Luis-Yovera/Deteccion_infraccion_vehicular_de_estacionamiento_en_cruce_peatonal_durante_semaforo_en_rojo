import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame= cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 90, 50])
    upper = np.array([60, 220, 180])

    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()