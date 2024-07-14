import cv2
import numpy as np
import pyautogui
import pyautogui as pag
import HandTracker as ht


class VirtualMouse:
    def __init__(self):
        # hand data
        self._handTracker = ht.HandTracker(handsAmount=1)
        self._fingerTips = [4, 8, 12, 16, 20]
        # camera data
        self._windowName = "Virtual Mouse"
        self._camera = cv2.VideoCapture(1)
        self._cameraWidth, self._cameraHeight = 640, 480
        self._camera.set(3, self._cameraWidth)
        self._camera.set(4, self._cameraHeight)
        self._screenWidth, self._screenHeight = pag.size()
        # mouse data
        self._trackpadMargin = 140
        self._lagFix = 1.2
        pyautogui.FAILSAFE = False

    def __del__(self):
        if self._camera.isOpened():
            self._camera.release()
        cv2.destroyAllWindows()

    def _whichMode(self, areFingersUp):
        if (not areFingersUp[0] and areFingersUp[1] and not areFingersUp[2]
                and not areFingersUp[3] and not areFingersUp[4]):
            return "Move"
        elif (areFingersUp[0] and areFingersUp[1] and not areFingersUp[2]
              and not areFingersUp[3] and not areFingersUp[4]):
            return "LeftClick"
        elif (not areFingersUp[0] and areFingersUp[1] and areFingersUp[2]
              and not areFingersUp[3] and not areFingersUp[4]):
            return "ScrollUp"
        elif (areFingersUp[0] and areFingersUp[1] and areFingersUp[2]
              and not areFingersUp[3] and not areFingersUp[4]):
            return "ScrollDown"
        elif (areFingersUp[0] and areFingersUp[1] and areFingersUp[2]
              and areFingersUp[3] and areFingersUp[4]):
            return "Break"
        else:
            return "Idle"

    def run(self):
        cv2.namedWindow(self._windowName, cv2.WINDOW_KEEPRATIO)
        currentX, currentY, previousX, previousY = 0, 0, 0, 0
        while cv2.getWindowProperty(self._windowName, cv2.WND_PROP_VISIBLE) >= 1:
            success, frame = self._camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            frame = cv2.flip(frame, 1)
            frame = self._handTracker.trackHands(frame)
            landmarks = self._handTracker.parsePosition(frame)
            cv2.rectangle(frame, (self._trackpadMargin, self._trackpadMargin),
                          (self._cameraWidth - self._trackpadMargin, self._cameraHeight - self._trackpadMargin),
                          (0, 0, 0), 2)
            if landmarks:
                indexX, indexY = landmarks[8]
                areFingersUp = self._handTracker.fingersPosition(landmarks, frame)
                if self._whichMode(areFingersUp) == "Move":
                    currentX = np.interp(indexX, (self._trackpadMargin, self._cameraWidth - self._trackpadMargin),
                                  (0, self._screenWidth))
                    currentY = np.interp(indexY, (self._trackpadMargin, self._cameraHeight - self._trackpadMargin),
                                  (0, self._screenHeight))
                    currentX = previousX + (currentX - previousX) / self._lagFix
                    currentY = previousY + (currentY - previousY) / self._lagFix
                    pag.moveTo(currentX, currentY, duration=0)
                    previousX, previousY = currentX, currentY
                elif self._whichMode(areFingersUp) == "LeftClick":
                    pag.click(button="left")
                elif self._whichMode(areFingersUp) == "ScrollUp":
                    pag.scroll(5)
                elif self._whichMode(areFingersUp) == "ScrollDown":
                    pag.scroll(-5)
                elif self._whichMode(areFingersUp) == "Break":
                    break

            cv2.imshow(self._windowName, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break


if __name__ == "__main__":
    vm = VirtualMouse()
    vm.run()
