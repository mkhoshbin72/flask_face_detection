from flask import  Flask, render_template, Response
import cv2




app = Flask(__name__)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        # frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

        # faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    return frame

def get_frame():

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detectAndDisplay(frame)
        ret_, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)