
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_CLASSES = 36
EPOCHS = 8
BATCH_SIZE = 128

# Target classes: 0-9 and A-Z
TARGET_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# EMNIST Balanced source labels:
# 0-9, A-Z, and 11 additional lowercase classes
SOURCE_CHARACTERS = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZABDEFGHNQRT"
)

# Map all 47 source labels into 36 target classes
LABEL_MAPPING = [
    TARGET_CHARACTERS.index(character)
    for character in SOURCE_CHARACTERS
]

label_mapping_tensor = tf.constant(
    LABEL_MAPPING,
    dtype=tf.int64
)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading EMNIST Balanced dataset...")

(train_data, test_data), info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully!")

# ==========================================
# 3. PREPROCESS DATASET
# ==========================================

def preprocess_image(image, label):

    # Convert image to float
    image = tf.cast(image, tf.float32) / 255.0

    # Ensure image has one channel
    image = tf.reshape(image, (28, 28, 1))

    # Convert 47 source labels into 36 target labels
    label = tf.gather(
        label_mapping_tensor,
        label
    )

    return image, label


print("Preparing training and testing data...")

train_data = train_data.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_data = test_data.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_data = train_data.shuffle(
    10000
).batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

test_data = test_data.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

# ==========================================
# 4. BUILD CNN MODEL
# ==========================================

print("Building CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nStarting handwriting model training...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

print("\nEvaluating handwriting model...")

test_loss, test_accuracy = model.evaluate(
    test_data
)

print("\nTraining completed!")
print("Test Loss:", test_loss)
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

model.save("handwriting_model.keras")

print("\nModel saved successfully!")
print("File: handwriting_model.keras")

# Save character mapping
with open("handwriting_labels.txt", "w") as file:
    file.write(TARGET_CHARACTERS)

print("Character labels saved successfully!")
