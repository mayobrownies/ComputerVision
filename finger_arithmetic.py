import cv2
import mediapipe as mp
import time

class FingerCounter:
  def __init__(self):
    self.hands = self.mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    self.drawing = mp.solutions.drawing_utils 

    # position of finger tips from mediapipe
    self.finger_tips = [8, 12, 16, 20] 
    self.thumb_tip = 4

    # create expression string, keep track of previous input type, and add a delay to get hand in place
    self.expression = ""
    self.last_input_type = None
    self.last_input_time = time.time()
    self.delay_time = 3

  def count_fingers(self, hand_landmarks):
    fingers_up = 0
    # counts a finger as up if the finger tip is above a lower joint
    for tip_idx in self.finger_tips:
      if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[tip_idx - 2].y:
        fingers_up += 1

    # separately keep track of the thumb since the x-coordinate is important
    thumb_tip = hand_landmarks.landmark[self.thumb_tip]
    thumb_joint = hand_landmarks.landmark[self.thumb_tip - 2]
    if thumb_tip.y < thumb_joint.y and thumb_tip.x < thumb_joint.x:
      fingers_up += 1

    return fingers_up

  def get_operator(self, hand_landmarks, handedness):
    # use left hand for operators
    if handedness == 'Left':
      # location of finger tips
      index_tip = hand_landmarks.landmark[8]
      middle_tip = hand_landmarks.landmark[12]
      ring_tip = hand_landmarks.landmark[16]
      pinky_tip = hand_landmarks.landmark[20]
      
      # for divison, check if index, middle, ring, and pinky fingers are up
      if index_tip.y < hand_landmarks.landmark[6].y and middle_tip.y < hand_landmarks.landmark[10].y and ring_tip.y < hand_landmarks.landmark[14].y and pinky_tip.y < hand_landmarks.landmark[18].y:
        return '/'

      # for multiplication, check if index, middle, and ring fingers are up
      elif index_tip.y < hand_landmarks.landmark[6].y and middle_tip.y < hand_landmarks.landmark[10].y and ring_tip.y < hand_landmarks.landmark[14].y:
        return '*'
      
      # for subtraction, check if index and middle fingers are up
      elif index_tip.y < hand_landmarks.landmark[6].y and middle_tip.y < hand_landmarks.landmark[10].y:
        return '-'
      
      # for addition, check if index finger is up
      elif index_tip.y < hand_landmarks.landmark[6].y:
        return '+'
    
    return None

  # run program
  def run(self):
    video = cv2.VideoCapture(0)
    size = 1500
    video.set(3, size)
    video.set(4, size)

    while True:
      isTrue, img = video.read()
      if not isTrue:
        print("image not read")
        break

      img = cv2.flip(img, 1)

      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      result = self.hands.process(rgb_img)

      if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
          handedness = hand_info.classification[0].label

          self.drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
          finger_count = self.count_fingers(hand_landmarks)

          current_time = time.time()
          if current_time - self.last_input_time >= self.delay_time:
            if handedness == 'Right':
              if self.last_input_type != 'number':
                self.expression += str(finger_count)
                self.last_input_type = 'number'
                self.last_input_time = current_time

            elif handedness == 'Left':
              operator = self.get_operator(hand_landmarks, handedness)
              if operator and self.last_input_type == 'number':
                self.expression += operator
                self.last_input_type = 'operator'
                self.last_input_time = current_time

          cv2.putText(img, f'fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

          cv2.putText(img, f'expression: {self.expression}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      cv2.imshow('finger counter', img)

      if cv2.waitKey(1) & 0xFF == ord('d'):
        try:
          result = eval(self.expression)
          print(result)
          break
        except Exception as e:
          print(e)
          break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    finger_counter = FingerCounter()
    finger_counter.run()
