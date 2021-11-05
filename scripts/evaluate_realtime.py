# %%

import sys
import time

import cv2

sys.path.append("../facenetLib")  # go to parent dir
import contributed.face as face


class DetectorFace(object):

    def __init__(self):
        self.image = None
        self.model = '../models/20180402-114759.pb'
        self.save_point = '../models/one_shot_classifier.pkl'
        self.color = (0, 255, 0)
        self.text_color = (255, 0, 0)
        self.rect_size = 2
        self.threshold = .5
        self.face_recognition = face.Recognition(self.model, self.save_point)

    def predict(self, frame):
        faces = self.face_recognition.identify(frame)

        if faces:
            for img in faces:
                face_bb = img.bounding_box.astype(int)
                pt1 = (face_bb[0], face_bb[1])
                pt2 = (face_bb[2], face_bb[3])
                print(img.prediction)
                if img.name is not None and img.prediction > self.threshold:
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


detecter = DetectorFace()

video_0_path = '../data/wisenet_dataset/video/set_3/video3_1.avi'
video_1_path = '../data/wisenet_dataset/video/set_3/video3_4.avi'
videos = []
videos.append(cv2.VideoCapture(video_0_path))
videos.append(cv2.VideoCapture(video_1_path))

frame_rate = 1
prev = 0
text_box = (0, 25)
text_color = (255, 0, 0)
font_scale = 1

try:
    while videos[0].isOpened() and videos[1].isOpened():
        time_elapsed = time.time() - prev
        for idx in range(2):
            ret, frame = videos[idx].read()
            if time_elapsed > 1. / frame_rate:
                prev = time.time()
                if ret:
                    begin_time = time.time()
                    frame = detecter.predict(frame)
                    end_time = time.time()
                    cv2.putText(frame,
                                "Mean of time detector: {:.2f}".format(end_time - begin_time),
                                text_box,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                text_color,
                                thickness=2,
                                lineType=2)
                    cv2.imshow('detector_{}'.format(idx), frame)
                else:
                    break
        if cv2.waitKey(20) == ord('q'):
            break

except KeyboardInterrupt:
    pass
except Exception as ex:
    print(ex)
finally:
    videos[0].release()
    videos[1].release()
    cv2.destroyAllWindows()
