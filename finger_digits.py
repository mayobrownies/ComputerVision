import cv2
import numpy as np

class FingerCounter:
  def __init__(self, width=500, height=500):
      self.video = cv2.VideoCapture(0)
      self.video.set(3, width)
      self.video.set(4, height)

  def count_fingers(self, img):
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      
      lower_skin = np.array([0, 20, 70], dtype=np.uint8)
      upper_skin = np.array([20, 255, 255], dtype=np.uint8)
      
      skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
      
      kernel = np.ones((3,3), np.uint8)
      skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
      skin_mask = cv2.dilate(skin_mask, kernel, iterations=4)
      
      contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      if not contours:
          cv2.putText(img, 'Fingers Up: 0', 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0, 255, 0), 2)
          return img
      
      hand_contour = max(contours, key=cv2.contourArea)
      
      hull = cv2.convexHull(hand_contour, returnPoints=False)
      
      defects = cv2.convexityDefects(hand_contour, hull)
      
      finger_count = 0
      
      if defects is not None:
          for i in range(defects.shape[0]):
              s, e, f, d = defects[i, 0]
              start = tuple(hand_contour[s][0])
              end = tuple(hand_contour[e][0])
              far = tuple(hand_contour[f][0])
              
              a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
              b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
              c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
              
              angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
              
              if angle < np.pi/2 and d > 10000:
                  finger_count += 1
                  cv2.circle(img, far, 8, [0, 0, 255], -1)
      
      cv2.drawContours(img, [hand_contour], 0, (0, 255, 0), 2)
      
      finger_count = min(finger_count + 1, 5)
      
      cv2.putText(img, f'finger count: {finger_count}', 
                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                  1, (0, 255, 0), 2)
      
      return img

  def run(self):
    while True:
        isTrue, img = self.video.read()
        if not isTrue:
            break
        
        img = cv2.flip(img, 1)
        
        img = self.count_fingers(img)
        
        cv2.imshow('finger counter', img)
        
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break
    
    self.video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    finger_counter = FingerCounter()
    finger_counter.run()