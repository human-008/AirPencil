import cv2
import numpy as np
cap=cv2.VideoCapture(0)
sensitivity=5
lower_bound=np.array([100+sensitivity,50,50])
upper_bound=np.array([130-sensitivity,255,255])
kernel = np.ones((3,3),np.uint8)
centres=[]
while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_bound,upper_bound)
    mask=cv2.erode(mask,kernel,iterations=2)
    for i in range(30):
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    mask=cv2.bilateralFilter(mask,9,75,75)
    mask=cv2.dilate(mask,kernel,iterations=2)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cntrs,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,cntrs,-1,(0,0,255),3)
    for contour in cntrs:
        (x,y,w,h) = cv2.boundingRect(contour)
        #(x,y), r = cv2.minEnclosingCircle(cntrs)
        if cv2.contourArea(contour)<400:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        pt = (int(x+w/2), int(y+h/2))
        centres.append(pt)
    if len(centres)>=2:
        for i in range(1, len(centres)):
            cv2.line(frame,centres[i-1],centres[i],(0,0,255),4)
    if len(centres)==25:
        centres=[]
        
    cv2.imshow('original',frame)
    cv2.imshow('result',res)
    k=cv2.waitKey(30)
    if k==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
    