
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PredictDetection:
    def __init__(self):
        self.bounding_box = None
        self.name = None
        self.prediction = None


class Net:
    def __init__(self, box=(224, 224, 3), trainable=False):
        self.box = box
        self.model = self.get_model(trainable)

    def get_model(self, trainable=False):
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.box, include_top=False, weights='imagenet')
        self.base_model.trainable = trainable

        model = tf.keras.Sequential([
            self.base_model,  # 1
            tf.keras.layers.Conv2D(32, 3, activation='relu'),  # 2
            tf.keras.layers.Dropout(0.2),  # 3
            tf.keras.layers.GlobalAveragePooling2D(),  # 4
            tf.keras.layers.Dense(5, activation='softmax')  # 5
        ])
        return model


class CNNRecognition:
    def __init__(self, img_width, img_height):
        from nets.detector_MTCNN import DetectorMTCNN
        self.detector = DetectorMTCNN(img_width, img_height)
        self.recognition = Net((img_width, img_height, 3))
        self.img_width = img_width
        self.img_height = img_height
        labels_path = '../models/labels.txt'
        self.imagenet_labels = np.array(open(labels_path).read().splitlines())

    def get_label(self, predictions):
        predicted_class = np.argmax(predictions[0])        
        predicted_class_name = self.imagenet_labels[predicted_class]
        return predicted_class_name

    def identify(self, image):
        faces = self.detector.face_detection(image)
        if faces:
            predict_list = []
            for face_box in faces:
                predict = PredictDetection()              
                x, y, width, height = face_box           
                predict.bounding_box = np.array([x, y, x+width, y+height])   
                face_image = image[y:y+height, x:x+width, :]            
                img_batch = np.expand_dims(face_image, axis=0)
                img_batch = tf.image.resize(img_batch, (self.img_width, self.img_height), method=ResizeMethod.BILINEAR)
                img_preprocessed = preprocess_input(img_batch)
                model_predict = self.recognition.predict(img_preprocessed)
                predict.prediction = np.max(model_predict[0])
                predict.name = self.get_label(model_predict)
                predict_list.append(predict)
            return predict_list
        return None
