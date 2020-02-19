import MAG
import scene
import argparse
import pickle

parser = argparse.ArgumentParser(description='Renders a trajectory.')
parser.add_argument("-n", type=int, default=0, help="number of scenes in dir")
parser.add_argument("-dir", type=str, default="./scenes", help="directory")
args = parser.parse_args()

values = {}
polis = {}

for i in range(args.n):
    s = scene.from_file(args.dir+"/scene"+str(i))
    for j in range(s.get_trajectory_length()):
        team_dict, e_dict = s.to_input_dicts(j)
        tar_dict = s.to_target_dict(j)
        tnodes = team_dict["nodes"]
        tnodes.flags.writeable = False
        enodes = e_dict["nodes"]
        enodes.flags.writeable = False
        key = tnodes.tostring()+enodes.tostring()
        values.setdefault(key, []).append(tar_dict["globals"])
        polis.setdefault(key, []).append(tar_dict["nodes"])

for v,p in zip(values.values(), polis.values()):
    print(v,p)
