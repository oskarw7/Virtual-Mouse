import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, mode=False, handsAmount=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self._handsResult = None
        self._handsAmount = handsAmount
        self._mpHands = mp.solutions.hands
        self._hands = self._mpHands.Hands(mode, handsAmount, modelComplexity, detectionConfidence, trackConfidence)
        self._mpDraw = mp.solutions.drawing_utils
        self._fingerTips = [4, 8, 12, 16, 20]

    def trackHands(self, frame):
        self._handsResult = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self._handsResult.multi_hand_landmarks:
            for hand in self._handsResult.multi_hand_landmarks:
                self._mpDraw.draw_landmarks(frame, hand, self._mpHands.HAND_CONNECTIONS)
        return frame

    def parsePosition(self, frame, handNumber=0):
        landmarks = []
        if self._handsAmount >= handNumber >= 0 and self._handsResult.multi_hand_landmarks:
            thisHand = self._handsResult.multi_hand_landmarks[handNumber]
            for _, landmark in enumerate(thisHand.landmark):
                height, width, channel = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append((x, y))
        return landmarks

    def whichHand(self, frame):
        if self._handsResult.multi_handedness and self._handsAmount == 1:
            return self._handsResult.multi_handedness[0].classification[0].label

    def fingersPosition(self, landmarks, frame):
        areFingersUp = []
        if not landmarks:
            return areFingersUp
        if self.whichHand(frame) == "Left":
            if landmarks[self._fingerTips[0]][0] > landmarks[self._fingerTips[0] - 1][0]:
                areFingersUp.append(True)
            else:
                areFingersUp.append(False)
        else:
            if landmarks[self._fingerTips[0]][0] < landmarks[self._fingerTips[0] - 1][0]:
                areFingersUp.append(True)
            else:
                areFingersUp.append(False)
        for fingerTip in self._fingerTips[1:]:
            if landmarks[fingerTip][1] < landmarks[fingerTip - 2][1]:
                areFingersUp.append(True)
            else:
                areFingersUp.append(False)
        return areFingersUp


def drawFPS(frame, previousTime):
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    return previousTime, frame


# used for launching tracker separately
# if you want to use this script as a module, omit this part or use it as a scheme
def main():
    windowName = "Hand Tracker"
    camera = cv2.VideoCapture(1)
    cv2.namedWindow(windowName, cv2.WINDOW_KEEPRATIO)
    handTracker = HandTracker(handsAmount=1)

    previousTime = 0
    while cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) >= 1:
        success, frame = camera.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = cv2.flip(frame, 1)
        frame = handTracker.trackHands(frame)
        landmarks = handTracker.parsePosition(frame)
        previousTime, frame = drawFPS(frame, previousTime)
        cv2.imshow(windowName, frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
