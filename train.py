import ray
import numpy as np
import copy

import tree_search
from config import config

load_latest = False


def make_inference_model():
    import model
    return model.InferenceModel(config) #init model with dimensions from examplenroll_trajectory(scene, config, predictor, replay_buffer)


#starts the training process
#scene gen must be a function with no arguments, that produces a fresh random scene on call
def start(scene_gen):

    ray.init()

    replay_buffer = tree_search.ReplayBuffer(config)

    def make_train_model():
        import model
        example = scene_gen()
        example.save_policy(np.zeros((len(example.actor_history[-1]), config["num_actions"]))) #save dummy policy to make dummy target
        return model.TrainableModel(config, *example.to_input_dicts(0), example.to_target_dict(0)) #init model with dimensions from examplenroll_trajectory(scene, config, predictor, replay_buffer)

    @ray.remote
    class Sampler(object):
        def __init__(self):
            example = scene_gen()
            example.save_policy(np.zeros((len(example.actor_history[-1]), config["num_actions"]))) #save dummy policy to make dummy target
            self.predictor = make_inference_model()

        def sample_once(self):
            scenes = []
            for _ in range(config["num_trajectories_per_thread"]):
                scene = scene_gen()
                tree_search.unroll_trajectory(scene, config, self.predictor)
                scenes.append(scene)
            return scenes

        def load_model(self):
            self.predictor.load_latest()

    # Create actors
    samplers = [Sampler.remote() for _ in range(config["threads"])]

    print("sampling random trajectories before training starts...")
    if load_latest:
        for s in samplers:
            s.load_model.remote()
    for scenes in ray.get([s.sample_once.remote() for s in samplers]):
        for s in scenes:
            replay_buffer.add(s)

    predictor = make_train_model()

    losses = []
    for epoch in range(config["epochs"]):
        print("Start training...")
        for i in range(config["train_iterations"]):
            batch_team_inputs, batch_enemy_inputs, batch_targets = replay_buffer.sample_batch(config["batch_size"])
            loss, ce, mse = predictor.train(batch_team_inputs, batch_enemy_inputs, batch_targets)
            losses.append(loss.numpy())
            print("epoch ", epoch, "it ", i, " loss=", losses[-1], " av100=", sum(losses[-100:])/min(100, len(losses)), " ce=", ce, " mse=", mse)
        predictor.save()
        print("Start resampling...")
        for s in samplers:
            s.load_model.remote()
        for scenes in ray.get([s.sample_once.remote() for s in samplers]):
            for s in scenes:
                replay_buffer.add(s)



def test(scene_gen, opponent, num_tests):

    ray.init()

    test_config = copy.deepcopy(config)
    #disable random components and go full exploit.
    test_config["random_moves_init"] = 0
    test_config["root_exploration_fraction"] = 0

    @ray.remote
    class Tester(object):
        def __init__(self):
            self.predictor = make_inference_model()
            self.predictor.load_latest()

        def test(self):
            scores = []
            for _ in range(num_tests):
                test_scene = scene_gen()
                while not test_scene.terminal():
                    actions_model, _ = tree_search.monte_carlo_search(test_config, test_scene, self.predictor)
                    test_scene.update(actions_model)
                    actions_opp = opponent(test_scene)
                    test_scene.update(actions_opp)
                scores.append(test_scene.score())
                print("...")
            return scores

    testers = [Tester.remote() for _ in range(test_config["threads"])]
    scores = []
    for score in ray.get([t.test.remote() for t in testers]):
        scores.extend(score)
    print("testing done with average score: ", sum(scores)/len(scores))
