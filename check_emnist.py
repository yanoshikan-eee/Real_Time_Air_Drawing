
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# EMNIST Balanced character mapping (47 classes)
# Labels 0-9: Digits
# Labels 10-35: Uppercase A-Z
# Labels 36-46: Additional lowercase classes converted to uppercase

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZABDEFGHNQRT"

# Load a sample of 1000 training images
dataset = tfds.load(
    "emnist/balanced",
    split="train[:1000]",
    as_supervised=True
)

# Randomly select 25 images
dataset = dataset.shuffle(
    buffer_size=1000,
    reshuffle_each_iteration=True
).take(25)

# Create the display window
plt.figure(figsize=(10, 10))

for i, (image, label) in enumerate(dataset):

    # Convert image and label
    image_array = image.numpy().squeeze()
    label_index = int(label.numpy())

    # Get corresponding character
    character = characters[label_index]

    # Display handwriting sample
    plt.subplot(5, 5, i + 1)

    plt.imshow(image_array, cmap="gray")
    plt.title(f"Character: {character}")
    plt.axis("off")

# Display final result
plt.suptitle(
    "EMNIST Handwriting Recognition - A to Z and 0 to 9",
    fontsize=14
)

plt.tight_layout()
plt.show()

print("Handwriting samples displayed successfully!")
print("Total character classes:", len(characters))
