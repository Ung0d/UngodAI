import MAG
import scene
import model
import tensorflow as tf
import numpy as np
from config import config
import argparse

parser = argparse.ArgumentParser(description='Checks predictions of the latest model and compares them with the actual policies and target values.')
parser.add_argument("-i", type=int, default=0, help="number of the trajectory")
parser.add_argument("-dir", type=str, default="./scenes", help="directory")
args = parser.parse_args()

model = model.InferenceModel(config)
model.load_latest()
scene = scene.from_file(args.dir+"/scene"+str(args.i))
for i in range(scene.get_trajectory_length()):
    team, enemies = scene.to_input_dicts(i)
    target = scene.to_target_dict(i)
    v,p = model.predict([team], [enemies])
    p = tf.nn.softmax(p).numpy()
    scene.print_info(i)
    print("predicted value=", v)
    print("predicted polies=", p)
    print("______________________________")
print("end: ", scene.score())
