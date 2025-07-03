import cv2
import numpy as np

#load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#analyze skin tone based on HSV hue values
def analyze_skin_tone(face_region):
    #convert to HSV
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

    #focus on central area of the face
    h, w, _ = hsv.shape
    center = hsv[h//4:h*3//4, w//4:w*3//4]

    #get hue channel
    hue = center[:, :, 0].flatten()
    avg_hue = np.mean(hue)

    #determine tone based on hue value
    if 5 <= avg_hue <= 35:
        return "Warm"
    elif 90 <= avg_hue <= 130:
        return "Cool"
    else:
        return "Neutral"

#open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        tone = analyze_skin_tone(face)

        #color recommendations
        if tone == "Warm":
            recommendation = "Try olive, mustard, coral, or cream colors."
            color = (0, 140, 255)  # orange
        elif tone == "Cool":
            recommendation = "Try emerald, navy, lavender, or grey tones."
            color = (255, 0, 255)  # purple
        else:
            recommendation = "Try mint, soft pink, grey, or white."
            color = (200, 200, 200)  # grey

        #draw face rectangle and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{tone} Tone", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, recommendation, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #show frame
    cv2.imshow('Skin Tone Analyzer', frame)

    #exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release camera and close windows
cap.release()
cv2.destroyAllWindows()