import numpy as np
import matplotlib.pyplot as plt
import cv2


cap = cv2.VideoCapture(0)

while True:
    _, frame= cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('primera app en tiempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()