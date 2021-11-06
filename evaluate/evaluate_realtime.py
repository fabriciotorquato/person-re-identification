import argparse
from threading import Thread
import cv2
import time
import os
import json
import sys
import datetime

from database import Tracker, createTable, getConnection, insert, select, update
sys.path.append("../facenetLib")  # go to parent dir
sys.path.append("..")  # go to parent dir
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from contributed import face

class DetectorFace(object):

    def __init__(self, set, id, hash):
        self.image = None
        self.model = '../models/20180402-114759.pb'
        self.save_point = '../models/one_shot_classifier.pkl'
        self.color = (0, 255, 0)
        self.text_color = (255, 0, 0)
        self.rect_size = 2
        self.threshold = .5
        self.face_recognition = face.Recognition(self.model, self.save_point)
        self.location = "space {}".format(id)
        self.json_results = {"set": set, "tracks": {}}
        self.hash = hash

    def insert_predict(self, predict_name, msec):
        con = getConnection()
        createTable(con)
        last_predict = select(con, self.hash, predict_name, self.location)
        
        if len(last_predict) > 0:
            last_predict = last_predict[0]
            last_predict = Tracker(*last_predict)
        else:
            last_predict = None
        
        if last_predict and msec - last_predict.end < 1000:
            update(con, last_predict.tracker_id, msec)
        else:
            insert(con, self.hash, predict_name, self.location, msec, msec)


    def predict(self, frame, timestamp):
        faces = self.face_recognition.identify(frame)
        if faces:
            for img in faces:
                face_bb = img.bounding_box.astype(int)
                pt1 = (face_bb[0], face_bb[1])
                pt2 = (face_bb[2], face_bb[3])
                if img.name is not None and img.prediction > self.threshold:
                    self.insert_predict(img.name, timestamp)
                    cv2.rectangle(frame, pt1, pt2, self.color, self.rect_size)
                    cv2.putText(frame,
                                img.name,
                                pt2,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                self.text_color,
                                thickness=2,
                                lineType=2)

        return frame


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, idx=0, frame=None):
        self.frame = frame
        self.stopped = False
        self.idx = idx

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.frame = cv2.resize(self.frame, (480, 270))
            cv2.imshow("Video_{}".format(self.idx), self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.datetime.now() -
                        self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0


def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.datetime.now() -
                        self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0


def noThreading(idx=0, source=0, detecter=None):
    """Grab and show video frames without multithreading."""

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        frame = detecter.predict(frame)
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video_{}".format(idx), frame)
        cps.increment()


def threadVideoGet(idx=0, source=0, detecter=None):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = detecter.predict(frame)
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()


def threadVideoShow(idx=0, source=0, detecter=None):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(idx, frame).start()
    cps = CountsPerSec().start()
    frame_rate = 2
    prev = 0

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break
        msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            frame = detecter.predict(frame, msec)
            frame = putIterationsPerSec(frame, cps.countsPerSec())
            video_shower.frame = frame
            cps.increment()


def threadBoth(idx=0, source=0, detecter=None):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(idx, video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = detecter.predict(frame)
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", "-x", default=0,
                    help="Number for sync many videos (default 0).")
    ap.add_argument("--video", "-v", default="/home",
                    help="Path to video file (default /home).")
    ap.add_argument("--set", "-s", default=0,
                    help="Set to video file (default 0).")
    ap.add_argument("--id", "-i", default=0,
                    help="Index to video file (default 0).")
    ap.add_argument("--thread", "-t", default="none",
                    help="Threading mode: get (video read in its own thread),"
                         + " show (video show in its own thread), both"
                         + " (video read and video show in their own threads),"
                         + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    source = '{}/set_{}/video{}_{}.avi'.format(
        args["video"], args["set"], args["set"], args["id"])
   

    if os.path.isfile(source):       
        detecter = DetectorFace(args["set"], args["id"], args["hash"])
        con = getConnection()
        createTable(con)

        if args["thread"] == "both":
            threadBoth(args["id"], source, detecter)
        elif args["thread"] == "get":
            threadVideoGet(args["id"], source, detecter)
        elif args["thread"] == "show":
            threadVideoShow(args["id"], source, detecter)
        else:
            noThreading(args["id"], source, detecter)


if __name__ == "__main__":
    main()
