import MAG
import scene
import argparse
import pickle

parser = argparse.ArgumentParser(description='Renders a trajectory.')
parser.add_argument("-i", type=int, default=0, help="number of the trajectory")
parser.add_argument("-dir", type=str, default="./scenes", help="directory")
args = parser.parse_args()

scene = scene.from_file(args.dir+"/scene"+str(args.i))
scene.draw_trajectory()
