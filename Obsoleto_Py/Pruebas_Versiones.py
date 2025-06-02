import tensorflow as tf
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))