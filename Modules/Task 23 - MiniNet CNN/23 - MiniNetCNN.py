# === 1. Import Libraries ===
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# === 2. Load and Preprocess MNIST Data ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape to include channel dimension (grayscale = 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values (0–255) → (0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# === 3. Define the MiniNet Architecture ===
MiniNet = models.Sequential([
    # --- Feature Extraction Block 1 ---
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # --- Classifier Head ---
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes (digits 0–9)
])

# === 4. Compile the Model ===
MiniNet.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# === 5. Early Stopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === 6. Train the Model ===
history = MiniNet.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === 7. Evaluate on Test Set ===
test_loss, test_acc = MiniNet.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# === 8. Plot Learning Curves ===
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MiniNet Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('MiniNet Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# === 9. Helper function to draw boxes ===
def draw_box(ax, text, xy, color, width=1.2, height=0.6):
    box = FancyBboxPatch(
        (xy[0], xy[1]), width, height,
        boxstyle="round,pad=0.02",
        linewidth=2, facecolor=color, edgecolor='black'
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2,
        text, fontsize=11, ha='center', va='center'
    )

# === Setup Figure ===
fig, ax = plt.subplots(figsize=(11, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# === Draw Layers ===
draw_box(ax, "Input\n28×28×1", (0.5, 2.0), "#A8DADC")

draw_box(ax, "Conv2D\n3×3×32", (2.0, 2.7), "#F4A261")
draw_box(ax, "Conv2D\n3×3×64", (2.0, 1.3), "#F4A261")
draw_box(ax, "MaxPooling\n2×2 → 12×12×64", (3.5, 2.0), "#2A9D8F")

draw_box(ax, "Flatten", (5.0, 2.8), "#E9C46A")
draw_box(ax, "Dense (64)\nReLU", (5.0, 1.6), "#E9C46A")
draw_box(ax, "Dropout (0.5)", (6.5, 2.8), "#E9C46A")
draw_box(ax, "Dense (128)\nReLU", (6.5, 1.6), "#E9C46A")
draw_box(ax, "Dense (10)\nSoftmax", (8.0, 2.0), "#E76F51")

# === Arrows for data flow ===
def arrow(x1, y1, x2, y2):
    ax.arrow(x1, y1, x2 - x1, y2 - y1,
             width=0.03, head_width=0.15, head_length=0.2,
             length_includes_head=True, color="black")

arrow(1.7, 2.3, 2.0, 2.3)
arrow(3.2, 2.3, 3.5, 2.3)
arrow(4.8, 2.3, 5.0, 2.3)
arrow(6.3, 2.3, 6.5, 2.3)
arrow(7.8, 2.3, 8.0, 2.3)

# === Title ===
ax.text(5, 4.5, "MiniNet CNN Architecture (Optimized for MNIST)", 
        fontsize=14, ha='center', weight='bold')

plt.show()
