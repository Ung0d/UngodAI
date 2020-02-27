import MAG
import train
import ray

ray.init()
train.start(lambda : MAG.make_scene(), lambda : MAG.make_fair_scene())
