import sys
sys.path.append("../facenetLib")
sys.path.append("..")

import tensorflow as tf
import cv2
import os

from evaluate.database import Tracker, createTable, getConnection, insert, select, update


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class AIDetector:

    def __init__(self, set, id, hash, face_recognition):
        self.image = None
        self.color = (0, 255, 0)
        self.text_color = (255, 0, 0)
        self.rect_size = 2
        self.threshold = .2
        self.face_recognition = face_recognition
        self.location = "space {}".format(id)
        self.json_results = {"set": set, "tracks": {}}
        self.hash = hash
        self.img_width, self.img_height = 224, 224

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
                print(img.name, img.prediction)
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


class CnnDetector(AIDetector):

    def __init__(self, set, id, hash):
        from nets.cnn_recognition import CNNRecognition
        model = '../models/fine_tuning'
        face_recognition = CNNRecognition(img_width=224, img_height=224)
        face_recognition.recognition = tf.keras.models.load_model(model)
        super().__init__(set, id, hash, face_recognition)


class FacenetDetector(AIDetector):

    def __init__(self, set, id, hash):
        model = '../models/20180402-114759.pb'
        save_point = '../models/one_shot_classifier.pkl'
        from contributed import face
        face_recognition = face.Recognition(model, save_point)
        super().__init__(set, id, hash, face_recognition)


def get_detector(name):
    if name == "facenet":
        return FacenetDetector
    elif name == "cnn":
        return CnnDetector
