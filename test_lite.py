import numpy as np
import tensorflow as tf
import sonnet as snt


def make_mlp_model():
    return snt.nets.MLP([50, 50, 50], activate_final=False)

model = make_mlp_model()

inference = tf.function(lambda x: model(x), input_signature=[tf.TensorSpec(shape=(None, 50), dtype=tf.float32)])

converter = tf.lite.TFLiteConverter.from_concrete_functions([inference.get_concrete_function()])
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
