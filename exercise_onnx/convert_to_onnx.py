# convert_to_onnx.py
import tf2onnx
import tensorflow as tf
import onnx

# Load the Keras model
model = tf.keras.models.load_model("exercise_model.h5")


# Ensure the model is built by running a dummy prediction
model.predict(tf.random.uniform((1,) + model.input_shape[1:]))

# Define the input signature for the model
spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32),)

# Set the output names
model.output_names = [f"output_{i}" for i in range(len(model.outputs))]

# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save the ONNX model to a file
onnx.save(onnx_model, "path_to_save_model.onnx")

# with open("exercise_model.onnx", "wb") as f:
#    f.write(onnx_model.SerializeToString())
