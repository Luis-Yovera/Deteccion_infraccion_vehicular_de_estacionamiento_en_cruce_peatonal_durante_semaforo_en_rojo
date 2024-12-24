import cv2
import numpy as np

cap = cv2.VideoCapture(0)

selem1 = np.ones((5, 5), np.uint8)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 100, 100])
    upper = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, selem1)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, selem1)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, selem1)

    res = cv2.bitwise_and(frame, frame, mask=closing)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)
    cv2.imshow('Gradient', gradient)
    cv2.imshow('Segmentacion', res)

    # Cambiar la tecla para salir
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):  # Salir si se presiona 'q'
        break

cv2.destroyAllWindows()
cap.release()
