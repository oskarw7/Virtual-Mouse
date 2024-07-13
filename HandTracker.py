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
            for index, landmark in enumerate(thisHand.landmark):
                height, width, channel = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append((index, x, y))
        return landmarks


def drawFPS(frame, previousTime):
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    return previousTime, frame


def main():
    windowName = "Hand Tracker"
    captured = cv2.VideoCapture(1)
    cv2.namedWindow(windowName, cv2.WINDOW_KEEPRATIO)
    handTracker = HandTracker(handsAmount=1)

    previousTime = 0
    while cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) >= 1:
        success, frame = captured.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = cv2.flip(frame, 1)
        frame = handTracker.trackHands(frame)
        landmarks = handTracker.parsePosition(frame)
        if landmarks:
            print(landmarks)
        previousTime, frame = drawFPS(frame, previousTime)
        cv2.imshow(windowName, frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
