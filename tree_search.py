import numpy as np
import tensorflow as tf
import time
import copy
import os

class Node:

    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, logits, valid):
        policy = {i:np.math.exp(p) for i,(p,v) in enumerate(zip(logits, valid)) if v}
        div = sum(policy.values())
        self.children = {i:Node(p/div) for i,p in policy.items()}

    def apply_noise(self, config):
        #print([child_node.prior for child_node in self.children.values()])
        noise = np.random.gamma(config["root_dirichlet_alpha"], 1, len(self.children))
        frac = config["root_exploration_fraction"]
        for child_node, n in zip(self.children.values(), noise):
            child_node.prior = child_node.prior * (1 - frac) + n * frac

    def select_child(self, config):
        #select child nodes based on high value and low visit counts
        _, action, next = max([(ucb_score(config, self, child), action, child)
                               for action, child in self.children.items()])
        return action, next


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child):
    pb_c = np.log((parent.visit_count + config["pb_c_base"] + 1) /
                  config["pb_c_base"]) + config["pb_c_init"]
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


#simultaneously returns an action for the player and for each enemy, based on
#a monte carlo tree search run
def monte_carlo_search(config, scene, predictor, cache):

    #initialize search by creating roots and expanding them (+ apply exploration noise)
    team1, team2 = scene.get_team_and_enemies(-1)
    roots1 = [Node() for actor in team1]
    roots2 = [Node() for actor in team2]
    evaluate_and_expand(config, scene, roots1, predictor, cache, apply_noise=True)
    #iteratively and simultaneously construct search trees for team and enemies
    for i in range(config["tree_simulations"]):
        scene_local = scene.clone()
        search_paths1 = [[root] for root in roots1]
        search_paths2 = [[root] for root in roots2]
        #simi = time.process_time()
        turn = True
        search_paths_turn = search_paths1
        #according to ucb_score, traverse down all trees until at least one team is completely at leafs
        while (not scene_local.terminal() and
                not all([path[-1].is_leaf() for path in search_paths_turn])):
            actions = []
            for path in search_paths_turn:
                if not path[-1].is_leaf():
                    action, child = path[-1].select_child(config)
                    actions.append(action)
                    path.append(child)
                else:
                    actions.append(-1)
            #update both teams subsequently
            scene_local.update(actions)
            turn = not turn
            search_paths_turn = search_paths1 if turn else search_paths2
        #print("sim:", time.process_time() - simi)
        #simi = time.process_time()
        #evaluate all leafs
        eval_paths = search_paths1+search_paths2 if turn else search_paths2+search_paths1
        diff_values = evaluate_and_expand(config, scene_local, [path[-1] for path in eval_paths], predictor, cache, apply_noise=False)
        backpropagate(eval_paths, diff_values)
        #print("sim2:", time.process_time() - simi)
    policies = np.concatenate([make_policy(root, config["num_actions"]) for root in roots1 if not root.is_leaf()], axis=0)
    return get_actions(config, scene, policies), policies


#traverse up the search path and update node values and counters
def backpropagate(search_paths, values):
    for p,v in zip(search_paths, values):
        for node in p:
            node.value_sum += v
            node.visit_count += 1


#computes a policy from visit counts
def make_policy(node, num_actions):
    poli = np.zeros((1,num_actions))
    sum_visits = sum(child.visit_count for child in node.children.values())
    for i, child in node.children.items():
        poli[0,i] = child.visit_count / sum_visits
    return poli


#depending of the current trajectory length, an action for the player
#and each enemy is sampled either randomly or deterministically
def get_actions(config, scene, policies):
    if scene.get_trajectory_length() < config["random_moves_init"]:
        return [np.random.choice(range(policy.shape[0]),size=1,p=policy) for policy in policies]
    else:
        return [np.argmax(policy) for policy in policies]


#evalutes the predictor and expands both all team and enemy trees
#additionally applies noise
def evaluate_and_expand(config, scene, nodes, predictor, cache, apply_noise):
    team_actors, enemy_actors = scene.get_team_and_enemies(-1)
    if not scene.terminal():
        num_team_alive, num_enemy_alive = scene.get_num_alive(-1)
        team_dict, enemy_dict = scene.to_input_dicts(-1)
        team_one_out_dicts = [scene.to_input_dicts(-1, team_leave_out=id)[0] for id,actor in enumerate(team_actors) if actor.alive()] if num_team_alive > 1 else []
        enemy_one_out_dicts = [scene.to_input_dicts(-1, enemy_leave_out=id)[0] for id,actor in enumerate(enemy_actors) if actor.alive()] if num_enemy_alive > 1 else []
        hash = scene.to_hash(-1)
        team_hashes = [scene.to_hash(-1, team_leave_out=id)[0] for id,actor in enumerate(team_actors) if actor.alive()] if num_team_alive > 1 else []
        enemy_hashes = [scene.to_hash(-1, enemy_leave_out=id)[0] for id,actor in enumerate(enemy_actors) if actor.alive()] if num_enemy_alive > 1 else []
        values, logits = predictor.predict([team_dict]+team_one_out_dicts+[team_dict]*len(enemy_one_out_dicts),
                                            [enemy_dict]*(len(team_one_out_dicts)+1) + enemy_one_out_dicts,
                                            [hash] + team_hashes + enemy_hashes, cache)
        if len(team_one_out_dicts) == 0:
            values.insert(1, 0)
        if len(enemy_one_out_dicts) == 0:
            values.append(0)
        valid = scene.valid_actions(-1)
        alive_nodes = [node for node,actor in zip(nodes, team_actors) if actor.alive()]
        for node, l, v in zip(alive_nodes, logits[0], valid):
            if node.is_leaf():
                node.expand(l, v)
                if apply_noise:
                    node.apply_noise(config)
        difference_values = []
        #compute difference values
        vi = 1
        for actor in team_actors:
            if actor.alive():
                difference_values.append(values[0] - values[vi]) #values[0] = predicted game outcome, values[vi] = predicted game outcome without the actor
                vi += 1
            else:
                difference_values.append(0)
        for actor in enemy_actors:
            if actor.alive():
                difference_values.append(-values[0] + values[vi]) #values[0] = predicted game outcome, values[vi] = predicted game outcome without the actor
                vi += 1
            else:
                difference_values.append(0)
        return difference_values
    else: #no need to evaluate
        return [scene.score() if a.alive() else 0 for a in team_actors] + [-scene.score() if a.alive() else 0 for a in enemy_actors]


class ReplayBuffer:

    def __init__(self, config, dir="./scenes"):
        self.buffer = []
        self.config = config
        self.counter = 0
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

    def add(self, scene):
        if len(self.buffer) >= self.config["buffer_size"]:
            self.buffer.pop(0)
        self.buffer.append(scene)
        scene.to_file(self.dir + "/scene" + str(self.counter))
        self.counter += 1

    def sample_batch(self, batch_size):
        move_sum = float(sum(scene.get_trajectory_length() for scene in self.buffer))
        scenes = np.random.choice(
             self.buffer,
             size=batch_size,
             p=[scene.get_trajectory_length() / move_sum for scene in self.buffer])
        scene_pos = [(scene, np.random.randint(scene.get_trajectory_length())) for scene in scenes]
        team_inputs = []
        enemy_inputs = []
        targets = []
        for scene, i in scene_pos:
            team, enemies = scene.to_input_dicts(i)
            team_inputs.append(team)
            enemy_inputs.append(enemies)
            targets.append(scene.to_target_dict(i))
        return team_inputs, enemy_inputs, targets


def trajectory_step(scene, config, predictor, cache):
    actions, poli = monte_carlo_search(config, scene, predictor, cache)
    scene.update(actions)
    scene.save_policy(poli)
    return not scene.terminal() and scene.get_trajectory_length() <= config["max_trajectory_length"]

#unrolls a single trajectory up to a terminal state
# def unroll_trajectory(scene, config, predictor, cache):
#     while not scene.terminal() and scene.get_trajectory_length() <= config["max_trajectory_length"]:
#         actions, poli = monte_carlo_search(config, scene, predictor, cache)
#         scene.update(actions)
#         scene.save_policy(poli)
