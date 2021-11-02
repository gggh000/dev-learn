import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow import keras
import numpy as np
import onnxruntime

#model = ResNet50(weights='imagenet')
model = keras.models.load_model("p297.h5")
import tf2onnx
import onnxruntime as rt

#spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
spec = (tf.TensorSpec((None, 28, 28), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
