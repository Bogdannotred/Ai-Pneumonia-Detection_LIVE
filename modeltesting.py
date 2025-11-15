import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("best_model.h5")


IMG_SIZE = (224, 224)  

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "for_train_dataset/test/",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary', 
    shuffle=False     
)

# Evaluate model
results = model.evaluate(test_generator, return_dict=True)

print("\n===== Evaluation Results =====")
for k, v in results.items():
    print(f"{k}: {v:.6f}")

# If accuracy exists
if "accuracy" in results:
    print("Error rate:", 1 - results["accuracy"])