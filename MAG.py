import scene
import train
import numpy as np
import copy

map_x = 7
map_y = 7
graph_edge_min_dist = 100


num_action = 5 #up, down, left, right, pass

num_actor_per_team = 2


class MAGActor(scene.Actor):

    def __init__(self, px, py, score = 0, on_goal = False, is_alive = True):
        self.px = px
        self.py = py
        self.score = score
        self.is_alive = is_alive
        self.on_goal = on_goal


    def get_position(self):
        return np.array([self.px, self.py], dtype=np.float32)


    # def get_state(self):
    #     return np.array([self.px, self.py, self.on_goal, self.score], dtype=np.float32)

    #discrete grid space
    def get_state(self):
        state = np.zeros((map_x, map_y), dtype=np.float32)
        state[self.px, self.py] = 1
        return state


    def alive(self):
        return self.is_alive


    def update(self, action, actor_scene):

        if action == 0: #up
            self.py += 1
            if self.py >= map_y:
                self.py -= 1
        elif action == 1: #down
            self.py -= 1
            if self.py < 0:
                self.py += 1
        elif action == 2: #right
            self.px += 1
            if self.px >= map_x:
                self.px -= 1
        elif action == 3: #left
            self.px -= 1
            if self.px < 0:
                self.px += 1

        for i in range(actor_scene.additional_data.shape[0]):
            if (actor_scene.additional_data[i,0] == self.px and
                actor_scene.additional_data[i,1] == self.py):
                self.on_goal = True
                if actor_scene.additional_data[i,2] == 1: #first time visit
                    self.score = 1
                    actor_scene.additional_data[i,2] = 0
                break

        if not any(actor_scene.additional_data[:,2] == 1):
            self.is_alive = False



    def fitness(self):
        return np.float32(self.score)


    def clone(self):
        return MAGActor(self.px, self.py, self.score, self.on_goal, self.is_alive)


    def valid_actions(self, i, scene):
        return np.array([self.is_alive]*(num_action-1)+[1], dtype=np.float32)


    def to_hash(self):
        return self.px*map_y + self.py



goals = np.array([[2,2,1], [4,1,1]])

def make_scene():
    return scene.Scene(graph_edge_min_dist, map_x, map_y, num_actor_per_team,
                                    [[MAGActor(np.random.randint(map_x), np.random.randint(map_y)) for _ in range(2*num_actor_per_team)]],
                                    copy.deepcopy(goals))

#both teams start at the exact same positions
def make_fair_scene():
    actors = [MAGActor(np.random.randint(map_x), np.random.randint(map_y)) for _ in range(num_actor_per_team)]
    return scene.Scene(graph_edge_min_dist, map_x, map_y, num_actor_per_team,
                                    [actors + copy.deepcopy(actors)],
                                    copy.deepcopy(goals))
