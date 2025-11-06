import numpy as np
import cv2


def preprocess_image_onnx(image):
	"""
	Preprocess the input image for ONNX model prediction

	Args:
		image: numpy array of the image (RGB)

	Returns:
		Preprocessed image numpy array ready for ONNX inference
	"""
	# Resize to 128x128
	img = cv2.resize(image, (128, 128))

	# Ensure RGB format
	if len(img.shape) == 2:  # Grayscale
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	elif img.shape[2] == 4:  # RGBA
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

	# Normalize to [0, 1]
	img = img.astype(np.float32) / 255.0

	# Transpose to (C, H, W) format
	img = np.transpose(img, (2, 0, 1))

	# Add batch dimension (1, C, H, W)
	img_array = np.expand_dims(img, axis=0)

	return img_array