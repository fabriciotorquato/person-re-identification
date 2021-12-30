import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('../../libs/facenet/src')
sys.path.append('../../src')

import random
import datetime
import cv2
from threading import Thread
import argparse

from nets.ai_detector import get_detector
from re_indentification.database import createTable, getConnection

class VideoShow:

    def __init__(self, idx=0, frame=None):
        self.frame = frame
        self.stopped = False
        self.idx = idx
        self.changed = True

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def change_frame(self, frame):
        self.frame = frame
        self.changed = True

    def show(self):
        while not self.stopped:
            if self.changed:
                self.changed = False
                self.frame = cv2.resize(self.frame, (480, 270))
                cv2.imshow('Video_{}'.format(self.idx), self.frame)
            if cv2.waitKey(1) == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True


class CountsPerSec:

    def __init__(self, random_frames, fps):
        self._start_time = None
        self.num_occurrences = 0
        self.frame_count = 0
        self.random_frames = random_frames
        self.fps = fps
        self.list_random_frames = []
        self.reset()

    def start(self):
        self._start_time = datetime.datetime.now()
        return self

    def increment(self):
        self.num_occurrences += 1
        self.frame_count += 1

    def reset(self):
        self.num_occurrences = 0
        self.list_random_frames = random.sample(
            range(0, self.fps), self.random_frames)

    def is_frame(self):
        return self.num_occurrences in self.list_random_frames


def threadVideoShow(idx=0, source=0, detecter=None):

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_random_rate = 10
    video_shower = VideoShow(idx, frame).start()
    cps = CountsPerSec(frame_random_rate, fps).start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break
        msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        cps.increment()
        if msec % 1000 == 0:
            cps.reset()
        if cps.is_frame():
            frame = detecter.predict(frame, msec, cps.frame_count)
            video_shower.change_frame(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash', '-x', default=0,
                    help='Number for sync many videos (default 0).')
    ap.add_argument('--video', '-v', default='/home',
                    help='Path to video file (default /home).')
    ap.add_argument('--set', '-s', default=0,
                    help='Set to video file (default 0).')
    ap.add_argument('--id', '-i', default=0,
                    help='Index to video file (default 0).')
    ap.add_argument('--detector', '-d', default='facenet',
                    help='Detector mode: facenet or mobilenet')
    args = vars(ap.parse_args())

    source = '{}/set_{}/video{}_{}.avi'.format(
        args['video'], args['set'], args['set'], args['id'])

    if os.path.isfile(source):
        con = getConnection()
        createTable(con)
        detector_hub = get_detector(args['detector'])
        detecter = detector_hub(args['set'], args['id'], args['hash'])
        threadVideoShow(args['id'], source, detecter)


if __name__ == '__main__':
    main()
