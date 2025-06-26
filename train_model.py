from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import os

# 💡 Path to dataset folder
train_dir = "data/asl_alphabet_train/asl_alphabet_train"

# 📏 Image and batch details
img_size = (64, 64)
batch_size = 32

# 📸 Image augmentation with improvements
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

# 🚂 Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 🧪 Validation generator
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🧠 Confirm class detection
print("\n✅ Classes detected:", train_generator.class_indices)

# 🧠 Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# 💥 Compile with label smoothing for better generalization
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# 💾 Model saving setup
if not os.path.exists("model"):
    os.makedirs("model")

checkpoint = ModelCheckpoint("model/asl_cnn_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# 🚀 Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[checkpoint],
    verbose=1
)

# 🫶 Final model save
model.save("model/asl_cnn.h5")
print("\n🎉 Final model saved as 'model/asl_cnn.h5'")
