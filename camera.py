import cv2 as cv
import time
import numpy as np

video =cv.VideoCapture(0)
face=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

a= 0

while True:
    check, frame = video.read()
    # print(check)
    # print(frame)
    #print(frame.ndim)
    frame1 = np.array(frame)  #copying to save the file without face detection rectangle
    img = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(img,
    scaleFactor=1.05,
    minNeighbors=5)
    for x,y,w,h in faces:
        frame1= cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow('camera',frame1)
    k=cv.waitKey(1)
    a+=1
    if k == ord('c'):
        cv.imwrite(('new\img-' + time.strftime(
        '%Y-%m-%d-%H-%M-%S',time.localtime()) +'.jpg'),frame)

    if k == ord('x'):
        break



# print(video.getBackendName())


# cv.imwrite('new\capturedw.jpg',frame)
print('frames:'+str(a))
video.release()
cv.destroyAllWindows()
