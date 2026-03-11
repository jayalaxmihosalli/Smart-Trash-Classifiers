import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image


# ============================================
# Reproducibility
# ============================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ============================================
# Configuration
# ============================================

DATASET_PATH = "dataset"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(MODEL_DIR, "trash_model.h5")

CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25


# ============================================
# Create directories
# ============================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================
# GPU Configuration
# ============================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected")
else:
    print("Running on CPU")


# ============================================
# Data Generators
# ============================================

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

    subset="training",

    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)

class_names = list(train_generator.class_indices.keys())

print("Classes:", class_names)


# ============================================
# Save class names
# ============================================

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)


# ============================================
# Build CNN Model
# ============================================

model = Sequential([

    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512,activation='relu'),
    Dropout(0.5),

    Dense(256,activation='relu'),
    Dropout(0.3),

    Dense(len(class_names),activation='softmax')

])

model.summary()


# ============================================
# Compile Model
# ============================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================
# Callbacks
# ============================================

early_stop = EarlyStopping(

    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(

    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)


# ============================================
# Train Model
# ============================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint, reduce_lr]
)


# ============================================
# Save Model
# ============================================

model.save(MODEL_PATH)

print("Model saved successfully")


# ============================================
# Plot Accuracy
# ============================================

plt.figure()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

plt.savefig("accuracy.png")


# ============================================
# Plot Loss
# ============================================

plt.figure()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])

plt.savefig("loss.png")


# ============================================
# Evaluation
# ============================================

validation_generator.reset()

predictions = model.predict(validation_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = validation_generator.classes


# ============================================
# Classification Report
# ============================================

print(classification_report(y_true, y_pred, target_names=class_names))


# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(y_true, y_pred)

plt.figure()

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names,

    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")


# ============================================
# Prediction Function
# ============================================

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    return predicted_class, confidence


print("Training Complete")