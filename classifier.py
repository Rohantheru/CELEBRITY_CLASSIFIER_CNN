import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

img_dir=r'CELEBRITY_DATA'
Messi_images=os.listdir(r'CELEBRITY_DATA\lionel_messi')
Maria_images=os.listdir(r'CELEBRITY_DATA\maria_sharapova')
Roger_images=os.listdir(r'CELEBRITY_DATA\roger_federer')
Serena_images=os.listdir(r'CELEBRITY_DATA\serena_williams')
Virat_images=os.listdir(r'CELEBRITY_DATA\virat_kohli')

ds = []
label = []
img_size = (128,128)


for i , image_name in tqdm(enumerate(Messi_images),desc="MESSI"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        ds.append(np.array(image))
        label.append(0)

for i , image_name in tqdm(enumerate(Maria_images),desc="MARIA"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        ds.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(Roger_images),desc="ROGER"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        ds.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(Serena_images),desc="SERENA"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        ds.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(Virat_images),desc="VIRAT"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        ds.append(np.array(image))
        label.append(4)

ds=np.array(ds)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(ds))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(ds,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation="softmax")
])

print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=35,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred_class = np.argmax(y_pred,axis = 1)
print('classification Report\n',classification_report(y_test,y_pred_class))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    labels = ["MESSI","MARIA","ROGER","SERENA","VIRAT"]
    prediction = np.argmax(labels)
    print("Predication: ",labels[prediction])
        

make_prediction(r'CELEBRITY_DATA\virat_kohli\virat_kohli10.png',model)
print("--------------------------------------\n")
