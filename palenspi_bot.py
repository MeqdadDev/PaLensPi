from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from credentials import BOT_TOKEN, BOT_USERNAME
import cv2 as cv
from detectors_world import DetectorCreator
from picamera2 import Picamera2
import math
import numpy as np


def findAngle(lmlist, img, p1, p2, p3, draw=True):
    _, x1, y1 = lmlist[p1]
    _, x2, y2 = lmlist[p2]
    _, x3, y3 = lmlist[p3]

    # Calc angle
    angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle +=360
    # print(angle)
    if draw:
        cv.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
        cv.line(img, (x3,y3), (x2,y2), (255,255,255), 3)
        cv.circle(img, (x1,y1), 8, (0,0,255), cv.FILLED)
        cv.circle(img, (x1,y1), 12, (0,0,255), 2)
        cv.circle(img, (x2,y2), 8, (0,0,255), cv.FILLED)
        cv.circle(img, (x2,y2), 12, (0,0,255), 2)
        cv.circle(img, (x3,y3), 8, (0,0,255), cv.FILLED)
        cv.circle(img, (x3,y3), 12, (0,0,255), 2)
        cv.putText(img, str(int(angle)), (x2-50, y2+50), cv.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    return angle

def checkPose(lmlist, img, p, ctr):
    _, x, y = lmlist[p]
    elbow_l = lmlist[13][1]

    cv.circle(img, (x, y), 8, (255, 0, 255), cv.FILLED)
    cv.putText(img, str(ctr), (40, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    if elbow_l < 400:
        cv.line(img, (400, 20), (400, 400), (255, 0, 255), 4)
    else:
        cv.line(img, (400, 20), (400, 400), (0, 255, 0), 4)

    # Define a state variable to track the position of the head
    # 0: head behind the line
    # 1: head crossed the line
    state = 0 if x <= 400 else 1

    # Increment the counter only when the state changes from 0 to 1
    global prev_state
    if state == 1 and state != prev_state:
        ctr += 1

    # Store the current state for the next iteration
    prev_state = state

    return ctr

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}, Welcome to {BOT_USERNAME} \n \
                                    By: PaLensPi Team.')

async def exercise1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cv.startWindowThread()
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    creator = DetectorCreator()
    pose = creator.getDetector("pose")
    count = 0
    direction = 0
    while True:
        rgb = picam2.capture_array()
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

        bgr = pose.detect(bgr, drawOnPose=True)
        landmarks = pose.locate(bgr)
        # print(landmarks)
        if len(landmarks) != 0:
            angle = findAngle(landmarks ,bgr, 12, 14, 16 ,True)
            per = np.interp(angle, (200, 320), (0,100))
            if per == 100:
                if direction == 0:
                    count+=0.5
                    direction = 1
            if per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0
            print('count::',count)
            cv.putText(bgr, f'{int(count)}', (45,400), cv.FONT_HERSHEY_PLAIN, 10, (255,0,0), 15)
            cv.putText(bgr, f'{int(per)} %', (75,130), cv.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        cv.imshow("LensPi - Exercise 1", rgb)

        k = cv.waitKey(1)
        if k % 255 == 27:
            break
    cv.destroyAllWindows()

async def exercise2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cv.startWindowThread()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    creator = DetectorCreator()
    pose = creator.getDetector("pose")

    c = 0
    prev_state = None
    while True:
        rgb = picam2.capture_array()
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        bgr = pose.detect(bgr, drawOnPose=True)
        landmarks = pose.locate(bgr)

        if len(landmarks) != 0:
            c = checkPose(landmarks, bgr, 8, c)

        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        cv.imshow("LensPi - Exercise 2", rgb)

        k = cv.waitKey(1)
        if k % 255 == 27:
            break
    cv.destroyAllWindows()



if __name__ == "__main__":

    app = ApplicationBuilder().token(f"{BOT_TOKEN}").build()

    app.add_handler(CommandHandler("hi", hello))
    app.add_handler(CommandHandler("ex1", exercise1))
    app.add_handler(CommandHandler("ex2", exercise2))


    app.run_polling()