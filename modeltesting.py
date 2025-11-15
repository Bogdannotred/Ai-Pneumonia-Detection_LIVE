import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("pneumonia_model.h5")

# Image size your model was trained on
IMG_SIZE = (224, 224)   # change if you used something else

# Test generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "for_train_dataset/test/",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',   # or 'categorical' if it was multi-class
    shuffle=False          # DO NOT shuffle for evaluation
)

# Evaluate model
results = model.evaluate(test_generator, return_dict=True)

print("\n===== Evaluation Results =====")
for k, v in results.items():
    print(f"{k}: {v:.6f}")

# If accuracy exists
if "accuracy" in results:
    print("Error rate:", 1 - results["accuracy"])