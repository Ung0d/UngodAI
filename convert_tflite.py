import model
import train
import MAG

model = train.make_inference_model(lambda : MAG.make_scene())
model.load_latest()
model.to_tflite()
