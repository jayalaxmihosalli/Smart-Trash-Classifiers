import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -------------------------------
# Configuration
# -------------------------------

DATASET_PATH = "dataset"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 25

MODEL_PATH = "model/trash_model.h5"


# -------------------------------
# Data Generator
# -------------------------------

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training"
)

validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation"
)

class_names = list(train_generator.class_indices.keys())

print("Classes:", class_names)


# -------------------------------
# Build CNN Model
# -------------------------------

model = Sequential()

# Layer 1
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Layer 2
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Layer 3
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Layer 4
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layers
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

# Output
model.add(Dense(len(class_names),activation='softmax'))

model.summary()

# -------------------------------
# Compile Model
# -------------------------------

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# -------------------------------
# Callbacks
# -------------------------------

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True
)

# -------------------------------
# Train Model
# -------------------------------

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]
)

# -------------------------------
# Save Model
# -------------------------------

model.save(MODEL_PATH)

print("Model saved!")

# -------------------------------
# Plot Accuracy
# -------------------------------

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train","val"])

plt.show()

# -------------------------------
# Plot Loss
# -------------------------------

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train","val"])

plt.show()