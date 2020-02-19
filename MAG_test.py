import MAG
import train
import numpy as np

def decide(scene):
    #move each actor towards a different goal
    team, _ = scene.get_team_and_enemies(-1)
    goals = zip(scene.additional_data, [False]*scene.additional_data.shape[0])
    actions = []
    for actor in team:
        if not actor.alive():
            actions.append(-1)
        else:
            dist = [(np.linalg.norm(g[:2]-actor.get_position()), g) for g, used in goals if not used]
            if len(dist) > 0:
                _, target = min(dist, key=lambda x: x[0])
                tx = target[0]
                ty = target[1]
                ax = actor.get_position()[0]
                ay = actor.get_position()[1]
                if tx > ax:
                    actions.append(2)
                elif tx < ax:
                    actions.append(3)
                elif ty > ay:
                    actions.append(0)
                elif ty < ay:
                    actions.append(1)
                else:
                    actions.append(4)
            else:
                #walk randomly
                actions.append(np.random.randint(MAG.num_action))
    return actions


train.test(lambda : MAG.make_fair_scene(), lambda scene : decide(scene), 100)
