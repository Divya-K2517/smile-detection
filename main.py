import cv2
import time
import numpy as np


cam = cv2.VideoCapture(0)
time.sleep(4)
#default frame width
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cam.read()
    #writes the frame to the output file
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detects faces
    cv2.putText(frame, "press q to exit", (int(frame_width*0.65), int(frame_height*0.85)), fontFace=3, fontScale=1.5, color=(0,0,255), thickness=3)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
        #roi = Region of Interest
        #x and y represent the top left corner coordinates of the roi, which is the face
        #h and w represent the height and width of the roi
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 30, minSize=(245,245))

        for (sx, sy, sw, sh) in smiles:
            pts = []
            # Adjust these values to move the curve up or down
            curve_height = int(sh * 0.5)  # Reduce the height of the curve
            curve_y_offset = int(sh * 0.1)  # Move the curve down

            for i in range(sw):
                px = x + sx + i
                # creating a parabolic curve
                py = y + sy + curve_y_offset + int(curve_height * (1 - ((i - sw/2)/(sw/2))**2))
                pts.append([px, py])

            cv2.polylines(frame, [np.array(pts, np.int32).reshape((-1, 1, 2))], isClosed=False, color=(0,255,0), thickness=3)

        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.1, minNeighbors=5)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(roi_color, (int(ex+ew/2), int(ey+eh/2)), int(ew/2), (0,255,0),3)

        if len(smiles) > 0: #if at least one smile is found
            cv2.putText(frame, "smile detected :))", (int(frame_width*0.1), int(frame_height*0.1)), fontFace=3, fontScale=1.5, color=(0,0,255), thickness=3)
        elif len(smiles) == 0 and len(eyes) > 0: #"no smile detected" will only appear if there is no smile and eyes, ensuring that the face is a true positive face
            cv2.putText(frame, "no smile detected :((", (int(frame_width*0.1), int(frame_height*0.1)), fontFace=3, fontScale=1.5, color=(0,0,255), thickness=3)

    cv2.imshow('smile detection', frame)
    #if you press q the loop will break
    if cv2.waitKey(1) == ord('q'):
        break
#releasing the camera
cam.release()
cv2.destroyAllWindows()