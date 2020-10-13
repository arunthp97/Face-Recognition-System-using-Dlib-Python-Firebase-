import dlib
import cv2
import urllib.request
import numpy as np

url='http://192.168.0.100:8080/shot.jpg'
face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

def scale_faces(face_rects, down_scale=1.5):
    faces = []
    for face in face_rects:
        scaled_face = dlib.rectangle(int(face.left() * down_scale),
                                    int(face.top() * down_scale),
                                    int(face.right() * down_scale),
                                    int(face.bottom() * down_scale))
        faces.append(scaled_face)
    return faces

def detect_faces(image, down_scale=1.5):
    image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale,
                              interpolation=cv2.INTER_LINEAR)
    faces = face_detector(image_scaled, 0)
  #  faces = [face.rect for face in faces]
    faces = scale_faces(faces, down_scale)
    return faces

if __name__ == "__main__":
    while(True):
        imgResp=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        image=cv2.imdecode(imgNp,-1)
        faces = detect_faces(image, down_scale=0.5)
        
        for face in faces:
            x,y,w,h = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x,y), (w,h), (255,200,150), 2)
    
        cv2.imshow("Image", image)
        if cv2.waitKey(1)==27:
            break;
    cv2.destroyAllWindows()
