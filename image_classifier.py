import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("breed_classifier.h5")

# Load and preprocess the image
img = cv2.imread("test.webp")
img_resized = cv2.resize(img, (128, 128))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)


img_ready = np.expand_dims(img_rgb, axis=0)

# Visualize the image
plt.figure()
plt.imshow(img_rgb)
plt.show()

# Make a prediction
pred = model.predict(img_ready)
print(pred)

predicted_class = np.argmax(pred, axis=1)

print(f"Predicted class: {predicted_class}")
