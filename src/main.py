import cv2 as cv

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()
