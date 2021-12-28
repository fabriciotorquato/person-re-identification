import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
from threading import Thread
import cv2
import sys
import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('../../libs/facenet/src')
sys.path.append('../../src')



from re_indentification.database import createTable, getConnection
from nets.ai_detector import get_detector

class VideoGet:
    '''
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    '''

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
    '''
    Class that continuously shows a frame using a dedicated thread.
    '''

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
            cv2.imshow('Video_{}'.format(self.idx), self.frame)
            if cv2.waitKey(1) == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec):
    '''
    Add iterations per second text to lower-left corner of a frame.
    '''

    cv2.putText(frame, '{:.0f} iterations/sec'.format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


class CountsPerSec:
    '''
    Class that tracks the number of occurrences ('counts') of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    '''

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
    '''Grab and show video frames without multithreading.'''

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord('q'):
            break

        frame = detecter.predict(frame)
        cv2.imshow('Video_{}'.format(idx), frame)
        cps.increment()


def threadVideoGet(idx=0, source=0, detecter=None):
    '''
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    '''

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord('q')) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = detecter.predict(frame)
        cv2.imshow('Video', frame)
        cps.increment()


def threadVideoShow(idx=0, source=0, detecter=None):
    '''
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    '''

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
        frame = detecter.predict(frame, msec)
        video_shower.frame = frame
        cps.increment()



def threadBoth(idx=0, source=0, detecter=None):
    '''
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    '''

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
        video_shower.frame = frame
        cps.increment()


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
    ap.add_argument('--thread', '-t', default='none',
                    help='Threading mode: get (video read in its own thread),'
                         + ' show (video show in its own thread), both'
                         + ' (video read and video show in their own threads),'
                         + ' none (default--no multithreading)')
    ap.add_argument('--detector', '-d', default='facenet',   help='Detector mode: facenet or mobilenet')
    args = vars(ap.parse_args())

    source = '{}/set_{}/video{}_{}.avi'.format(
        args['video'], args['set'], args['set'], args['id'])

    if os.path.isfile(source):
        con = getConnection()
        createTable(con)

        detector_hub = get_detector(args['detector'])
        detecter = detector_hub(args['set'], args['id'], args['hash'])



        if args['thread'] == 'both':
            threadBoth(args['id'], source, detecter)
        elif args['thread'] == 'get':
            threadVideoGet(args['id'], source, detecter)
        elif args['thread'] == 'show':
            threadVideoShow(args['id'], source, detecter)
        else:
            noThreading(args['id'], source, detecter)


if __name__ == '__main__':
    main()
