import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = "dataset"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(MODEL_DIR, "trash_model.h5")

CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 25


# ==========================================================
# Create directories if not exist
# ==========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# GPU Configuration (Optional)
# ==========================================================

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected")
    except RuntimeError as e:
        print(e)
else:
    print("Running on CPU")


# ==========================================================
# Data Generators
# ==========================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

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

print("Detected classes:", class_names)


# ==========================================================
# Save class labels
# ==========================================================

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)


# ==========================================================
# Build CNN Model
# ==========================================================

model = Sequential()

# Layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
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

# Output Layer
model.add(Dense(len(class_names),activation='softmax'))

model.summary()


# ==========================================================
# Compile Model
# ==========================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================================
# Callbacks
# ==========================================================

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


# ==========================================================
# Train Model
# ==========================================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint, reduce_lr]
)


# ==========================================================
# Save Model
# ==========================================================

model.save(MODEL_PATH)

print("Model saved successfully")


# ==========================================================
# Plot Accuracy
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train","Validation"])

plt.show()


# ==========================================================
# Plot Loss
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train","Validation"])

plt.show()


# ==========================================================
# Evaluation
# ==========================================================

validation_generator.reset()

predictions = model.predict(validation_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = validation_generator.classes


# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names
    )
)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names,

    cmap="Blues"
)

plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.title("Confusion Matrix")

plt.show()


# ==========================================================
# Prediction Function (for Streamlit)
# ==========================================================

from tensorflow.keras.preprocessing import image

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    return predicted_class, confidence


print("\nTraining Complete")