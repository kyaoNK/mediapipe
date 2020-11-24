import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="mediapipe/models/face_landmark.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details()[0]["shape"])
print(interpreter.get_input_details()[0]["dtyoe"])

print(interpreter.get_output_details()[0]["shape"])
print(interpreter.get_output_details()[0]["dtype"])