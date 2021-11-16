
from mtcnn.mtcnn import MTCNN
from os.path import isfile, join, exists
from pathlib import Path
import glob
import cv2
from tqdm import tqdm

class DetectorMTCNN:
    def __init__(self,img_width, img_height):
        self.detector = MTCNN()
        self.img_width= img_width
        self.img_height=img_height

    def crop_faces(self,img):
        faces = self.detector.detect_faces(img)
        if len(faces) == 0:
            return None
        x, y, width, height = faces[0]['box']
        return img[y:y+height, x:x+width, :]
    
    def face_detection(self,img):
        faces = self.detector.detect_faces(img)
        if len(faces) == 0:
            return None
        box_face_detection = []
        for face in faces:
            box_face_detection.append(face['box'])
        return box_face_detection
    
    def crop_align(self,data_dir, data_dir_align):
        if not exists(data_dir_align):
            for img_path in tqdm(glob.glob(data_dir+'/**/*.png')):
                if isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = self.crop_faces(img)
                    if  img is not None:
                        dim = (self.img_width, self.img_height)    
                        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_path=img_path.replace(data_dir, data_dir_align)
                        Path("/".join(img_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(img_path,img)
