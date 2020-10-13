from tkinter import *
from tkinter import messagebox
import os


a=os.getcwd()
if os.path.exists(a+"\\dataset\\")==False:
        os.mkdir(a+"\\dataset\\")


def close():
        global off
        off=True

def dataset():
        import dlib
        import cv2
        import numpy as np
        import urllib.request
        import os

        detector = dlib.get_frontal_face_detector()
        url=enter1.get()
        a=os.getcwd()
        name=enter2.get()
        roll_no=enter3.get()
        branch=enter4.get()
        gender=enter5.get()
        if len(entry1.get()) == 0:
                messagebox.showinfo("Alert","Socket is Empty.")
        elif len(entry2.get()) == 0 or len(entry3.get()) == 0 or len(entry4.get()) == 0 or len(entry5.get()) == 0 :
                messagebox.showinfo("Alert","Fileds Are Empty.")
        else:
                
                while(True):
                        imgResp=urllib.request.urlopen(url)
                        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                        img=cv2.imdecode(imgNp,-1)
                        cv2.imshow('image', img)
                        if cv2.waitKey(1)==27:
                            if os.path.exists(a+"\\dataset\\"+str(roll_no)+"\\")==False:
                                os.mkdir(a+"\\dataset\\"+str(roll_no)+"\\")
                                cv2.imwrite("dataset/"+str(roll_no)+"/"+str(branch)+"."+str(gender)+"."+str(name)+".jpg",img)  
                                entry2.delete(0,END)
                                entry3.delete(0,END)
                                entry4.delete(0,END)
                                entry5.delete(0,END)
                                break
                            else:
                                messagebox.showinfo("Alert","Already Exist.")
                                entry2.delete(0,END)
                                entry3.delete(0,END)
                                entry4.delete(0,END)
                                entry5.delete(0,END)
                                break

                cv2.destroyAllWindows()
            
    
    
def train():
        from face_embeddings import extract_face_embeddings
        from face_detector import detect_faces
        from db import add_embeddings
        import dlib

        shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
        face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

        def enroll_face(image, label,label1,label2,label3,label4,
                            embeddings_path="face_embeddings.npy",
                            labels_path="labels.cpickle",
                            labels_path1="labels1.cpickle",
                            labels_path2="labels2.cpickle",
                            labels_path3="labels3.cpickle",
                            labels_path4="labels4.cpickle",down_scale=1.0):

                faces = detect_faces(image, down_scale)
                if len(faces)<1:
                    return False
                if len(faces)>1:
                    raise ValueError("Multiple faces not allowed for enrolling")
                face = faces[0]
                face_embeddings = extract_face_embeddings(image, face, shape_predictor,
                                                          face_recognizer)
                add_embeddings(face_embeddings, label,label1,label2,label3,label4, embeddings_path=embeddings_path,
                               labels_path=labels_path)
                return True

        if __name__ == "__main__":
                
                import cv2
                import glob
                import argparse

                ap = argparse.ArgumentParser()
                ap.add_argument("-d","--dataset",default="dataset")
                ap.add_argument("-e","--embeddings",default="face_embeddings.npy")
                ap.add_argument("-l","--labels",default="labels.cpickle")
                ap.add_argument("-l1","--labels1",default="labels1.cpickle")
                ap.add_argument("-l2","--labels2",default="labels2.cpickle")
                ap.add_argument("-l3","--labels3",default="labels3.cpickle")
                ap.add_argument("-l4","--labels4",default="labels4.cpickle")

                args = vars(ap.parse_args())
                filetypes = ["png", "jpg"]
                dataset = args["dataset"].rstrip("/")
                imPaths = []
               
                for filetype in filetypes:
                    imPaths += glob.glob("{}/*/*.{}".format(dataset, filetype))
                sum=0
                int=1
                
                for path in imPaths:
                    
                    label= path.split("\\")[1]
                    s= path.split("\\")[2]
                    label2,label3,label4,i=s.split(".")
                    sum=sum+int
                    label1=sum
                    
                    image = cv2.imread(path)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    enroll_face(rgb, label,label1,label2,label3,label4, embeddings_path=args["embeddings"],
                                labels_path=args["labels"],
                                labels_path1=args["labels1"],
                                labels_path2=args["labels2"],
                                labels_path3=args["labels3"],
                                labels_path4=args["labels4"])
      

def recognize():
        if os.path.exists(a+"\\face_embeddings.npy") and os.path.exists(a+"\\labels.cpickle") and os.path.exists(a+"\\labels1.cpickle") and os.path.exists(a+"\\labels2.cpickle") and os.path.exists(a+"\\labels3.cpickle") and os.path.exists(a+"\\labels4.cpickle"):
                from threading import Thread

                if len(entry1.get()) == 0:
                        messagebox.showinfo("Alert","Socket is Empty.")
                else:
                        class A(Thread):
                                def run(self):
                                        import numpy as np
                                        import datetime

                                        
                                        now=datetime.datetime.now()
                                        def recognize_face(embedding, embeddings, labels,labels1,labels2,labels3,labels4, threshold=0.5):
                                                distances = np.linalg.norm(embeddings - embedding, axis=1)
                                                argmin = np.argmin(distances)
                                                minDistance = distances[argmin]
                                            
                                                if minDistance>threshold:
                                                    label = ""
                                                else:
                                                    label = labels[argmin]
                                            
                                                return (label, minDistance)
                            
                                        if __name__ == "__main__":
                                                
                                                import cv2
                                                import argparse
                                                from face_embeddings import extract_face_embeddings
                                                from face_detector import detect_faces
                                                import cloudpickle as cPickle
                                                import dlib
                                                import urllib.request
                                                import os

                                                global off
                                                off=False
                                                a=os.getcwd()
                                                url=enter1.get()
                                                ap = argparse.ArgumentParser()
                                                ap.add_argument("-i","--image", default="4.jpg")
                                                ap.add_argument("-e","--embeddings" ,default="face_embeddings.npy")
                                                ap.add_argument("-l", "--labels",default="labels.cpickle")
                                                ap.add_argument("-l1", "--labels1",default="labels1.cpickle")
                                                ap.add_argument("-l2", "--labels2",default="labels2.cpickle")
                                                ap.add_argument("-l3", "--labels3",default="labels3.cpickle")
                                                ap.add_argument("-l4", "--labels4",default="labels4.cpickle")
                                                args = vars(ap.parse_args())
                                            
                                                embeddings = np.load(args["embeddings"])
                                                labels = cPickle.load(open(args["labels"],"rb"))
                                                labels1 = cPickle.load(open(args["labels1"],"rb"))
                                                labels2 = cPickle.load(open(args["labels2"],"rb"))
                                                labels3 = cPickle.load(open(args["labels3"],"rb"))
                                                labels4 = cPickle.load(open(args["labels4"],"rb"))
                                                shape_predictor = dlib.shape_predictor("models/"
                                                                                       "shape_predictor_5_face_landmarks.dat")
                                                face_recognizer = dlib.face_recognition_model_v1("models/"
                                                                                                 "dlib_face_recognition_resnet_model_v1.dat")
                                                
                                                while(not off):
                                                    imgResp=urllib.request.urlopen(url)
                                                    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                                                    image=cv2.imdecode(imgNp,-1)
                                                    image_original = image.copy()
                                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                                
                                                    faces = detect_faces(image)
                                                    for face in faces:
                                                        embedding = extract_face_embeddings(image, face, shape_predictor, face_recognizer)
                                                        label = recognize_face(embedding, embeddings, labels,labels1,labels2,labels3,labels4)
                                                        label1 = recognize_face(embedding, embeddings, labels1,labels,labels2,labels3,labels4)
                                                        label2 = recognize_face(embedding, embeddings, labels2,labels,labels1,labels3,labels4)
                                                        label3 = recognize_face(embedding, embeddings, labels3,labels,labels1,labels2,labels4)
                                                        label4 = recognize_face(embedding, embeddings, labels4,labels,labels1,labels2,labels3)
                                                        (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
                                                        cv2.rectangle(image_original, (x1, y1), (x2, y2), (255, 120, 120), 2)
                                                        cv2.putText(image_original, label4[0], (x1, y1 - 10),
                                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                                        file1=open(a+"\\firebase1.txt","w")
                                                        file2=open(a+"\\firebase2.txt","w")
                                                        file3=open(a+"\\firebase3.txt","w")
                                                        file4=open(a+"\\firebase4.txt","w")
                                                        file5=open(a+"\\firebase5.txt","w")
                                                        file6=open(a+"\\firebase6.txt","w")
                                                        file1.write(str(label1[0]))
                                                        file2.write(label[0])
                                                        file3.write(label4[0])
                                                        file4.write(label2[0])
                                                        file5.write(label3[0])
                                                        file6.write(now.strftime("%Y-%m-%d "+"%H:%M:%S"))
                                                        file1.close()
                                                        file2.close()
                                                        file3.close()
                                                        file4.close()
                                                        file5.close()
                                                        file6.close()
                                                    cv2.imshow("Image", image_original)
                                                    cv2.waitKey(1)
                                                    if off==True:
                                                            break
                                                        
                                                        
                                                            
                                                            
                                                cv2.destroyAllWindows()
                                       
                            
                                

                            
                        class B(Thread):
                                def run(self):
                                        from firebase import firebase
                                        import os
        
                                        global off
                                        off=False
                                        firebase=firebase.FirebaseApplication('https://loginfirebase-44a0b.firebaseio.com/')
                                        a=os.getcwd()
                                        while(not off):
                                                if os.path.exists(a+"\\firebase1.txt") and os.path.exists(a+"\\firebase2.txt") and os.path.exists(a+"\\firebase3.txt") and os.path.exists(a+"\\firebase4.txt") and os.path.exists(a+"\\firebase5.txt") and os.path.exists(a+"\\firebase6.txt"):
                                                    if os.stat(a+"\\firebase1.txt").st_size!=0 and os.stat(a+"\\firebase2.txt").st_size!=0 and os.stat(a+"\\firebase3.txt").st_size!=0 and os.stat(a+"\\firebase4.txt").st_size!=0 and os.stat(a+"\\firebase5.txt").st_size!=0 and os.stat(a+"\\firebase6.txt").st_size!=0 :
                                                        file1=open(a+"\\firebase1.txt","r")
                                                        file2=open(a+"\\firebase2.txt","r")
                                                        file3=open(a+"\\firebase3.txt","r")
                                                        file4=open(a+"\\firebase4.txt","r")
                                                        file5=open(a+"\\firebase5.txt","r")
                                                        file6=open(a+"\\firebase6.txt","r")
                                                        firebase.put('Data',file1.read(),{"name":file3.read(),"roll_no":file2.read(),"branch":file4.read(),"gender":file5.read(),"time":file6.read()})            
                                                        file1=open(a+"\\firebase1.txt","w")
                                                        file2=open(a+"\\firebase2.txt","w")
                                                        file3=open(a+"\\firebase3.txt","w")
                                                        file4=open(a+"\\firebase4.txt","w")
                                                        file5=open(a+"\\firebase5.txt","w")
                                                        file6=open(a+"\\firebase6.txt","w")
                                                        file1.write("")
                                                        file2.write("")
                                                        file3.write("")
                                                        file4.write("")
                                                        file5.write("")
                                                        file6.write("")          
                                                if off==True:
                                                        if os.path.exists(a+"\\firebase1.txt") and os.path.exists(a+"\\firebase2.txt") and os.path.exists(a+"\\firebase3.txt") and os.path.exists(a+"\\firebase4.txt") and os.path.exists(a+"\\firebase5.txt") and os.path.exists(a+"\\firebase6.txt"):
                                                            file1.close()
                                                            file2.close()
                                                            file3.close()
                                                            file4.close()
                                                            file5.close()
                                                            file6.close()
                                                            os.remove(a+"\\firebase1.txt")
                                                            os.remove(a+"\\firebase2.txt")
                                                            os.remove(a+"\\firebase3.txt")
                                                            os.remove(a+"\\firebase4.txt")
                                                            os.remove(a+"\\firebase5.txt")
                                                            os.remove(a+"\\firebase6.txt")
                                                            break
                                                        


                                    
                                   


                        t1=A()
                        t2=B()

                        t1.start()
                        t2.start()
        else:
                messagebox.showinfo("Alert","Train data not found.")


win=Tk()
win.geometry("470x450")
win.resizable(width=False, height=False)
win.title("Face Recognition")
win.iconbitmap(r'icon.ico')

enter1=StringVar()
enter2=StringVar()
enter3=StringVar()
enter4=StringVar()
enter5=StringVar()

label=Label(win,text="Attendence System Using Face Recognition",font=("times 12 bold underline"),fg="red")
label.grid(row=1,column=2)

label1=Label(win,text="Socket : ",font=30,fg='Red')
label1.grid(row=2,column=1,sticky=E)
label2=Label(win,text="Name : ",font=30,fg='Blue')
label2.grid(row=3,column=1,sticky=E)
label3=Label(win,text="Enrollment No. : ",font=30,fg='Blue')
label3.grid(row=4,column=1,sticky=E)
label4=Label(win,text="Branch : ",font=30,fg='Blue')
label4.grid(row=5,column=1,sticky=E)
label5=Label(win,text="Gender : ",font=30,fg='Blue')
label5.grid(row=6,column=1,sticky=E)

entry1=Entry(win,textvariable=enter1,justify='left')
entry1.grid(row=2,column=2)
entry2=Entry(win,textvariable=enter2,justify='left')
entry2.grid(row=3,column=2)
entry3=Entry(win,textvariable=enter3,justify='left')
entry3.grid(row=4,column=2)
entry4=Entry(win,textvariable=enter4,justify='left')
entry4.grid(row=5,column=2)
entry5=Entry(win,textvariable=enter5,justify='left')
entry5.grid(row=6,column=2)

button1=Button(win,text="DataSet",command=dataset)
button1.grid(row=7,column=2)

button2=Button(win,text="Training",command=train)
button2.grid(row=8,column=1)

button3=Button(win,text="Recognize",command=recognize)
button3.grid(row=9,column=2)

label11=Label(win,text="")
label11.grid(row=10,column=2)
closebutton=PhotoImage(file="test.png")

button4=Button(win,image=closebutton,command=close)
button4.grid(row=11,column=2)

label6=Label(win,text="Minor Project Members :-",font=("Calibar",10),fg="blue")
label6.grid(row=12,column=1)

label7=Label(win,text="1.) Arun Kumar Thapa",font=("Calibar",10),fg="blue")
label7.grid(row=13,column=1,sticky=E)

label8=Label(win,text="2.) Arshil Singh Bhatia",font=("Calibar",10),fg="blue")
label8.grid(row=14,column=1,sticky=E)

label9=Label(win,text="3.) Arpit Jain",font=("Calibar",10),fg="blue")
label9.grid(row=15,column=1,sticky=E)

label10=Label(win,text="4.) Akshada Godbole",font=("Calibar",10),fg="blue")
label10.grid(row=16,column=1,sticky=E)

label12=Label(win,text="0875CS161032",font=("Calibar",10),fg="blue")
label12.grid(row=13,column=2)

label12=Label(win,text="0875CS161031",font=("Calibar",10),fg="blue")
label12.grid(row=14,column=2)

label12=Label(win,text="0875CS161029",font=("Calibar",10),fg="blue")
label12.grid(row=15,column=2)

label12=Label(win,text="0875CS161016",font=("Calibar",10),fg="blue")
label12.grid(row=16,column=2)

win.mainloop()
