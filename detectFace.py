import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)  # Use 0 for default camera, 1 for the second camera etc.

while True:
    successful_frame_read, frame = webcam.read()
    
    if not successful_frame_read:
        break

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w ,h) in face_coordinates: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    cv2.imshow('FaceDetect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop on 'q' key press
        break

print("Code Complete")

webcam.release()  # Release the webcam
cv2.destroyAllWindows()
