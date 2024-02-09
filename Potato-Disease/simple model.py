#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=30


# In[3]:


dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "plantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names=dataset.class_names
class_names


# In[5]:


len(dataset)


# In[6]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())


# In[7]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch[0].shape)


# In[72]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch[0].numpy())


# In[73]:


for image_batch,label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.title(class_names[label_batch[2]])
    plt.axis("off")


# In[74]:


plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[75]:


len(dataset)


# In[76]:


train_size=0.8
len(dataset)*train_size


# In[77]:


train_ds = dataset.take(54)


# In[78]:


test_ds=dataset.skip(54) #[:54]
len(test_ds)


# In[79]:


val_size=0.1
len(dataset)*val_size


# In[80]:


val_ds=test_ds.take(6)
len(val_ds)


# In[81]:


test_ds=test_ds.skip(6)
len(test_ds)


# In[12]:


def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds,val_ds,test_ds


# In[13]:


train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)


# In[14]:


len(train_ds)


# In[15]:


len(val_ds)


# In[16]:


len(test_ds)


# In[17]:


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[88]:


resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[89]:


data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomContrast(factor=0.2),
    layers.experimental.preprocessing.RandomZoom(0.2)])


# In[90]:


# Apply data augmentation to an image batch
for image_batch, label_batch in dataset.take(1):
    augmented_images = data_augmentation(image_batch)
    
    plt.figure(figsize=(10, 10))
    
    for i in range(9):  # Display the first 9 augmented images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    
    plt.show()


# In[91]:


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Assuming you have your dataset and class_names defined

# Load a single batch of images and labels from the dataset
for image_batch, label_batch in dataset.take(1):
    # Choose a single image from the batch
    image = image_batch[0]

    # Create a figure with subplots to display augmented images
    plt.figure(figsize=(7, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("Original Image")
    plt.axis("off")

    # Apply data augmentation: Random Flip
    flip_augmented = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(image)
    plt.subplot(2, 2, 2)
    plt.imshow(flip_augmented.numpy().astype("uint8"))
    plt.title("Random Flip")
    plt.axis("off")

    # Apply data augmentation: Random Rotation
    rotation_augmented = layers.experimental.preprocessing.RandomRotation(0.2)(image)
    plt.subplot(2, 2, 3)
    plt.imshow(rotation_augmented.numpy().astype("uint8"))
    plt.title("Random Rotation")
    plt.axis("off")

    # Apply data augmentation: Random Contrast
    contrast_augmented = layers.experimental.preprocessing.RandomContrast(factor=0.2)(image)
    plt.subplot(2, 2, 4)
    plt.imshow(contrast_augmented.numpy().astype("uint8"))
    plt.title("Random Contrast")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# In[92]:


input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape=input_shape)


# In[40]:


model.summary()


# In[94]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['acc'],
)


# In[95]:


history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)


# In[96]:


scores=model.evaluate(test_ds)


# In[97]:


scores


# In[98]:


history


# In[99]:


history.params


# In[100]:


history.history.keys()


# In[101]:


acc=history.history['acc']
val_acc=history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']


# In[102]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc,label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),loss,label='Training Loss')
plt.plot(range(EPOCHS),val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')


# In[18]:


import numpy as np
for images_batch,labels_batch in test_ds.take(1):
    first_image=images_batch[0].numpy().astype('uint8')
    first_label=labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("first image's actual label:",class_names[first_label])
    
    batch_prediction=model.predict(images_batch)
    print(class_names[np.argmax(batch_prediction[0])])


# In[104]:


print(np.argmax(batch_prediction[0]))


# In[10]:


def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,0)
    
    predictions=model.predict(img_array)
    print(predictions)
    
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence


# In[106]:


plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        
        plt.title(f"Actual:{actual_class},\n Predicted:{predicted_class},\n Confidence:{confidence}")
        plt.axis("off")


# In[107]:


model.save("simplemodel.h5")


# In[8]:


from numpy import loadtxt
from tensorflow.keras.models import load_model
 
# load model
model = load_model('simplemodel.h5')
# summarize model.
model.summary()


# In[3]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


# In[110]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report

# Assuming you have a trained model 'model' and a test dataset 'test_ds'
true_labels = []
predicted_labels = []

for image_batch, label_batch in test_ds:
    predictions = model.predict(image_batch)
    predicted_labels.extend(np.argmax(predictions, axis=-1))
    true_labels.append(label_batch.numpy())

true_labels = np.concatenate(true_labels)  # Convert the list to a NumPy array
predicted_labels = np.array(predicted_labels)

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heat Map')
plt.show()


# In[111]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(true_labels, predicted_labels))


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming you have a trained model 'model' and datasets 'train_ds', 'val_ds', 'test_ds'
true_labels = []
predicted_labels = []

for image_batch, label_batch in test_ds:
    predictions = model.predict(image_batch)
    predicted_labels.extend(np.argmax(predictions, axis=-1))
    true_labels.extend(label_batch.numpy().astype(int))

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print("Unique true labels:", np.unique(true_labels))
print("Unique predicted labels:", np.unique(predicted_labels))

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heat Map')
plt.show()


# In[ ]:




