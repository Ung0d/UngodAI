import abc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import graph_nets as gn
import time
import copy
import pickle


#an actor with a position and additional data
class Actor(abc.ABC):

    #returns the 2D position of the actor
    @abc.abstractmethod
    def get_position(self):
        pass

    #returns a np.array with all data describing the state of the actor
    @abc.abstractmethod
    def get_state(self):
        pass

    #performs the given action
    @abc.abstractmethod
    def update(self, action, scene):
        pass

    #returns a value between 0 (dead) and 1 (fully alive)
    @abc.abstractmethod
    def fitness(self):
        pass

    def alive(self):
        return self.fitness() > 0

    @abc.abstractmethod
    def to_hash(self):
        pass


    #returns a copy of the actor
    @abc.abstractmethod
    def clone(self):
        pass

    #returns a binary array indicating which actions are valid
    @abc.abstractmethod
    def valid_actions(self, i, scene):
        pass


def dist(actor1, actor2):
        return np.linalg.norm(actor1.get_position()-actor2.get_position())


class Scene(abc.ABC):

    def __init__(self, graph_edge_min_dist, map_x, map_y, team_1_size, actor_history, additional_data = None, policy_history = None):
        self.graph_edge_min_dist = graph_edge_min_dist
        self.map_x = map_x
        self.map_y = map_y
        self.team_1_size = team_1_size
        self.actor_history = actor_history
        self.policy_history = policy_history or []
        self.additional_data = additional_data


    #updates the scene performing the given actions, stores the policies in history
    def update(self, actions):
        clones = [actor.clone() for actor in self.actor_history[-1]]
        self.actor_history.append(clones)
        _, team = self.get_team_and_enemies(-1)
        for actor,action in zip(team, actions):
            if not action == -1 and actor.alive():
                actor.update(action, self)


    def save_policy(self, policy):
        self.policy_history.append(policy)


    def get_trajectory_length(self):
        return len(self.actor_history)-1


    def get_team_and_enemies(self, i):
        assert i == -1 or (i >= 0 and i < len(self.actor_history)), str(i) + " is not a valid index"
        team_index = (len(self.actor_history)-1)%2 if i == -1 else i%2
        enemy_index = 1 - team_index
        ts = [0,self.team_1_size,len(self.actor_history[i])]
        team = self.actor_history[i][ts[team_index]:ts[team_index+1]]
        enemies = self.actor_history[i][ts[enemy_index]:ts[enemy_index+1]]
        return team, enemies


    def get_num_alive(self, i = -1):
        team, enemies = self.get_team_and_enemies(i)
        return len([a for a in team if a.alive()]), len([a for a in enemies if a.alive()])


    def make_dict(self, actor_states):
        nodes = np.concatenate([np.reshape(s, (1,-1)) for s in actor_states], axis=0)
        nodes[:,0] /= self.map_x
        nodes[:,1] /= self.map_y
        senders = []
        receivers = []
        edges = []
        for x in range(len(actor_states)):
            for y in range(len(actor_states)):
                senders.append(x)
                receivers.append(y)
                edges.append([np.linalg.norm(nodes[x,:2] - nodes[y,:2])])

        return {"nodes": nodes,
                "globals": np.zeros(1, dtype=np.float32),
                "senders": senders,
                "receivers": receivers,
                "edges": edges}

    #i ... state index in trajectory (if even == team 1's turn, if odd == team 2's turn)
    #leave_out ... index in range(0, len(team)) optionally stating an actor to leave out
    #assumes that at least one actor in both team and enemies is alive
    #if leave_out is not None, at least one other actor than the one to leave out must be in team
    def to_input_dicts(self, i, team_leave_out = None, enemy_leave_out = None):
        team, enemies = self.get_team_and_enemies(i)
        team_alive = [actor.get_state() for j,actor in enumerate(team) if actor.alive() and not j == team_leave_out]
        enemies_alive = [actor.get_state() for j,actor in enumerate(enemies) if actor.alive() and not j == enemy_leave_out]
        return self.make_dict(team_alive), self.make_dict(enemies_alive)


    def to_target_dict(self, i):
        return {"nodes": self.policy_history[i],
                "globals": [self.score() if i%2==0 else -self.score()],
                "senders": [],
                "receivers": [],
                "edges": []}


    def to_hash(self, i, team_leave_out = None, enemy_leave_out = None):
        team, enemies = self.get_team_and_enemies(i)
        # team_hash = b"".join([a.get_state().tostring() for j,a in enumerate(team) if a.alive() and not j == team_leave_out])
        # enemy_hash = b"".join([a.get_state().tostring() for j,a in enumerate(enemies) if a.alive() and not j == enemy_leave_out])
        team_hash = tuple(a.to_hash() for j,a in enumerate(team) if a.alive() and not j == team_leave_out)
        enemy_hash = tuple(a.to_hash() for j,a in enumerate(enemies) if a.alive() and not j == enemy_leave_out)
        return (team_hash, enemy_hash)


    def terminal(self):
        return not any([e.alive() for e in self.actor_history[-1][:self.team_1_size]]) or not any([e.alive() for e in self.actor_history[-1][self.team_1_size:]])


    def score(self):
        return (sum(e.fitness() for e in self.actor_history[-1][:self.team_1_size])/self.team_1_size -
                sum(e.fitness() for e in self.actor_history[-1][self.team_1_size:])/len(self.actor_history[-1][self.team_1_size:]))


    def clone(self):
        return Scene(self.graph_edge_min_dist,
                     self.map_x, self.map_y,
                     self.team_1_size,
                     self.actor_history.copy(),
                     copy.deepcopy(self.additional_data),
                     self.policy_history.copy())


    def valid_actions(self, i):
        team, _ = self.get_team_and_enemies(i)
        return [e.valid_actions(i, self) for e in team if e.alive()]


    def print_info(self, i):
        print("score ", self.score())
        print("policies ", self.policy_history[i])
        print("turn ", 1-2*(i%2))
        team, enemies = self.get_team_and_enemies(i)
        for j, actor in enumerate(team):
            print("actor ", j, ":", actor.get_state())
        for j, actor in enumerate(enemies):
            print("enemy ", j, ":", actor.get_state())

    #renders an animation displaying the whole trajectory
    def draw_trajectory(self, secs_per_frame = 1):
        fig = plt.gcf()
        ax = plt.axes()
        fig.show()
        fig.canvas.draw()
        for i in range(self.get_trajectory_length()):
            ax.cla()
            ax.set_xlim(0, self.map_x)
            ax.set_ylim(0, self.map_y)
            self.draw_team_nodes(self.actor_history[i][:self.team_1_size], "r", ax)
            self.draw_team_nodes(self.actor_history[i][self.team_1_size:], "b", ax)
            fig.canvas.draw()
            self.print_info(i)
            time.sleep(secs_per_frame)

    def draw_team_nodes(self, team, color, ax):
        graph = nx.Graph()
        alive = [actor for actor in team if actor.alive()]
        for x,actor in enumerate(alive):
            graph.add_node(x, fitness="{:.2g}".format(actor.fitness()))
        nodepos = {x:actor.get_position() for x,actor in enumerate(alive)}
        node_fitness = {node: data["fitness"] for node, data in graph.nodes(data=True)}
        nx.draw(graph, pos=nodepos, node_color = color, node_size=50, ax=ax, labels = node_fitness)


    def to_file(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)


def from_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)
