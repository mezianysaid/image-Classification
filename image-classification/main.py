import tensorflow as tf 
import tf.keras as keras
import matplotlib.pyplot as plt 
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os

model_path='saved_model/train.ckpt' #  load the model from this path.
restore_model=keras.models.load_model(model_path)
weights=restore_model.get_weights()[0] # load weights
bias=restore_model.get_weights()[1]  # load biais

restore_model.summary()
def Test():
    image=keras.preprocessing.image.load_img(imageVar.get(),target_size=(180,180))
    img_array=keras.preprocessing.image.img_to_array(image) # Convert the image to an array
    img_array=tf.expand_dims(img_array,0)
    prediction=restore_model.predict(img_array)
#  prediction=restore_model.prediction(img)
    rslt=tf.nn.softmax(prediction[0])
    label=np.argmax(rslt)
    if label==0: 
        label="Cat"
    else: 
        label="Dog"
    print("label:{}".format(label))
    # label=np.argmax(prediction[0])
    plt.imshow(image)
    plt.title("Resultat: "+label)
    plt.axis('off')
    plt.show()
# ---------Interface -----------------------------------------------------------------------
window = tk.Tk()
window.geometry('800x500')
window.title("Reconnaissance d'image")
window.config(background="aqua")
left_frame = Frame(window, width=400, height=400)
left_frame.grid(row=0,column=0)
right_frame = Frame(window, width=400, height=400)
right_frame.grid(row=0,column=3,padx=30)


def show(event): 
    n=lst.curselection()
    imgname=lst.get(n)
    img=ImageTk.PhotoImage(Image.open(imgname))
    lab.config(image=img)
    lab.image=img
    imageVar.set(imgname)

images_url="data/test/"

def ListPaths():
    options=[]
    for i in os.listdir(images_url):
        full_path=images_url + i
        options.append(full_path)
    return options
list_images=ListPaths()
tk.Label(left_frame, text = "Selectionnez une image:",font="Helvetica  16 bold").pack()
lst=tk.Listbox(left_frame)
lst.pack(side="left",expand=1,fill=tk.BOTH)
# lst.config(background="grey")
# print(list_images)
for img in list_images: 
    lst.insert(tk.END,img)
img=ImageTk.PhotoImage(Image.open(list_images[4]))
lab=tk.Label(left_frame,image=img)
lab.pack()
imageVar=tk.StringVar(window) 
imageVar.set(list_images[4])
lst.bind("<<ListboxSelect>>",show)

button = Button(right_frame,command=Test, text="Test",fg = "white",bg="black",padx=1,pady=1,height=2, width=20,font="Helvetica  10 bold").pack()


window.mainloop()