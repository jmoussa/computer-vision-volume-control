import cv2
import time
import math
import numpy as np
import HandDetector as hd
import osascript

import platform


def main():
    if platform.system() != "Darwin":
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range = volume.GetVolumeRange()

    wCam, hCam, = (
        640,
        480,
    )
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = hd.HandDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    pTime = 0
    cTime = 0

    MAX_RANGE = 300
    vol_bar = 400
    vol_percent = 0
    while True:
        success, img = cap.read()
        # HAND TRACING
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list) > 0:
            thumb = lm_list[4]
            index_finger = lm_list[8]
            x1, y1 = thumb[1], thumb[2]
            x2, y2 = index_finger[1], index_finger[2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            if length > MAX_RANGE:
                MAX_RANGE = length

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            else:
                if platform.system() != "Darwin":
                    min_range, max_range = vol_range[0], vol_range[1]
                else:
                    min_range, max_range = 0, 100

                vol = np.interp(length, [50, MAX_RANGE], [min_range, max_range])
                vol_bar = np.interp(length, [50, MAX_RANGE], [400, 150])
                vol_percent = np.interp(length, [50, MAX_RANGE], [0, 100])

                if platform.system() != "Darwin":
                    volume.SetMasterVolumeLevel(vol, None)
                else:
                    osascript.osascript(f"set volume output volume {vol}")

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            f"{round(vol_percent, 2)}%",
            (40, 450),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            1,
        )

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (50, 80),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            3,
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
