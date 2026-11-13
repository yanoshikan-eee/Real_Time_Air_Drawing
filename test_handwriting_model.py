
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Character mapping
TARGET_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SOURCE_CHARACTERS = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZABDEFGHNQRT"
)

LABEL_MAPPING = [
    TARGET_CHARACTERS.index(character)
    for character in SOURCE_CHARACTERS
]

# Load trained model
print("Loading trained handwriting model...")

model = tf.keras.models.load_model(
    "handwriting_model.keras"
)

print("Model loaded successfully!")

# Load test dataset
print("Loading test images...")

dataset = tfds.load(
    "emnist/balanced",
    split="test",
    as_supervised=True
)

# Select 25 random test images
samples = list(
    dataset.shuffle(1000).take(25).as_numpy_iterator()
)

# Prepare images and labels
images = []
actual_labels = []

for image, label in samples:

    image = image.astype("float32") / 255.0
    image = image.reshape(28, 28, 1)

    mapped_label = LABEL_MAPPING[int(label)]

    images.append(image)
    actual_labels.append(mapped_label)

images = np.array(images)
actual_labels = np.array(actual_labels)

# Predict characters
print("Recognizing handwritten characters...")

predictions = model.predict(
    images,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

# Display results
plt.figure(figsize=(12, 10))

correct_count = 0

for i in range(25):

    actual = TARGET_CHARACTERS[actual_labels[i]]
    predicted = TARGET_CHARACTERS[predicted_labels[i]]

    confidence = np.max(predictions[i]) * 100

    if actual == predicted:
        correct_count += 1
        title_color = "green"
    else:
        title_color = "red"

    plt.subplot(5, 5, i + 1)

    plt.imshow(
        images[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted} ({confidence:.1f}%)",
        color=title_color,
        fontsize=9
    )

    plt.axis("off")

plt.suptitle(
    "Handwriting Recognition Test",
    fontsize=15
)

plt.tight_layout()
plt.show()

print("\nRecognition testing completed!")
print("Correct predictions:", correct_count, "/ 25")
print(
    "Sample accuracy:",
    round(correct_count / 25 * 100, 2),
    "%"
)
