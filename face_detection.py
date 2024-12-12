import cv2

width, height = 500, 500
video = cv2.VideoCapture(0)

video.set(3, width)
video.set(4, height)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
  isTrue, img = video.read()

  if not isTrue:
    print("failed to read frame")
    break

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = classifier.detectMultiScale(gray, scaleFactor=1.1)

  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

  cv2.imshow("video", img)
  if cv2.waitKey(1) & 0xFF == ord('d'):
    break

video.release()
cv2.destroyAllWindows()