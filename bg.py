import cv2 #importing computer vision

cap = cv2.VideoCapture(0)

#getting background image

while cap.isOpened():
    ret, background = cap.read()
    if ret:
        cv2.imshow("image",background)
        if cv2.waitKey(5) == ord('q'):
            cv2.imwrite("image.jpg",background)
            break
        #save the backaground
        
cap.release()
cv2.destroyAllWindows()
        