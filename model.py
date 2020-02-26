import numpy as np
import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import os

# SEED = 17
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

def make_mlp_model(layersizes):
    return lambda: snt.Sequential([
        snt.nets.MLP(layersizes, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])


#a graph NN that simultaneously predicts an appropriate action for each actor in a given scene,
#based on the current states of the respective actors and their topology
class ActionValuePredictor(snt.Module):

    def __init__(self, config):
        super(ActionValuePredictor, self).__init__(name="ActionValuePredictor")

        self.enemy_encoder = gn.modules.GraphIndependent(
                            edge_model_fn=make_mlp_model(config["encode_edge_layers"]),
                            node_model_fn=make_mlp_model(config["encode_node_layers"]),
                            global_model_fn=make_mlp_model(config["encode_global_layers"]))

        self.enemy_reducer = gn.modules.GraphNetwork(
                            edge_model_fn=make_mlp_model(config["enemy_reducer_edge_layers"]),
                            node_model_fn=make_mlp_model(config["enemy_reducer_node_layers"]),
                            global_model_fn=make_mlp_model(config["enemy_reducer_global_layers"]))

        self.team_encoder = gn.modules.GraphIndependent(
                            edge_model_fn=make_mlp_model(config["encode_edge_layers"]),
                            node_model_fn=make_mlp_model(config["encode_node_layers"]),
                            global_model_fn=make_mlp_model(config["encode_global_layers"]))

        self.core = gn.modules.GraphNetwork(
                            edge_model_fn=make_mlp_model(config["core_edge_layers"]),
                            node_model_fn=make_mlp_model(config["core_node_layers"]),
                            global_model_fn=make_mlp_model(config["core_global_layers"]))

        self.decoder = gn.modules.GraphIndependent(
                            edge_model_fn=make_mlp_model(config["encode_edge_layers"]),
                            node_model_fn=make_mlp_model(config["encode_node_layers"]),
                            global_model_fn=make_mlp_model(config["encode_global_layers"]))

        #initialize the final linear layer with 0 weights to ensure, that the initial prediction of the model is a uniform policy
        self.output_transform = gn.modules.GraphIndependent(
                            node_model_fn=lambda: snt.Linear(config["num_actions"], name="edge_output", w_init=snt.initializers.Constant(0), b_init=snt.initializers.Constant(0)),
                            global_model_fn=lambda: snt.Linear(1, name="global_output", w_init=snt.initializers.Constant(0), b_init=snt.initializers.Constant(0)))


    def __call__(self, team_input, enemy_input, steps):
        #first, learn a representation for the enemies, that is then forwarded as a global attribute to the team as globals
        enemy_reduced = self.enemy_reducer(self.enemy_encoder(enemy_input))
        team_input = team_input.replace(globals=enemy_reduced.globals)
        team_latent = self.team_encoder(team_input)
        team_latent0 = team_latent
        output_ops = []
        for _ in range(steps):
            core_input = gn.utils_tf.concat([team_latent0, team_latent], axis=1)
            team_latent = self.core(core_input)
            decoded_op = self.decoder(team_latent)
            output_ops.append(self.output_transform(decoded_op))
        return output_ops


def get_inputs(input_dicts):
    inputs = gn.utils_tf.data_dicts_to_graphs_tuple(input_dicts)
    return inputs


def get_targets(target_dicts):
    targets = gn.utils_tf.data_dicts_to_graphs_tuple(target_dicts)
    return targets


def make_loss(targets_tr, outputs_tr):
    ce_losses = tf.stack([tf.compat.v1.losses.softmax_cross_entropy(targets_tr.nodes, output.nodes)
                    for output in outputs_tr])
    mse_losses = tf.stack([tf.compat.v1.losses.mean_squared_error(targets_tr.globals, output.globals)
                    for output in outputs_tr])
    return ce_losses, mse_losses



class InferenceModel():

    def __init__(self, config, team_input_example, enemy_input_example):
        self.config = config
        self.model = ActionValuePredictor(config)
        self.checkpoint = tf.train.Checkpoint(module=self.model)
        self.checkpoint_root = "./checkpoints"
        self.checkpoint_name = "example"
        self.save_prefix = os.path.join(self.checkpoint_root, self.checkpoint_name)

        def inference(team_inputs_tr, enemy_inputs_tr):
            return self.model(team_inputs_tr, enemy_inputs_tr, config["test_mp_iterations"])

        team_input_example = get_inputs([team_input_example])
        enemy_input_example = get_inputs([enemy_input_example])

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
          gn.utils_tf.specs_from_graphs_tuple(team_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(enemy_input_example, dynamic_num_graphs=True)
        ]

        # Compile the update function using the input signature for speedy code.
        self.inference = tf.function(inference, input_signature=self.input_signature)


    def load_latest(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_root)
        if latest is not None:
            self.checkpoint.restore(latest)
            print("Loaded latest checkpoint")


    def save(self):
        self.checkpoint.save(self.save_prefix)
        print("Saved current model.")


    def to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_concrete_functions([self.inference.get_concrete_function()])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)



    #evaluations are cached
    #if a prediction for a reoccuring state is requested, time can be safed by just returning the cached values
    def predict(self, team_input_dicts, enemy_input_dicts, state_hashes, cache):
        is_cached = []
        ti_to_eval = []
        ei_to_eval = []
        for ti, ei, h in zip(team_input_dicts, enemy_input_dicts, state_hashes):
            is_cached.append(h in cache)
            if not is_cached[-1]:
                ti_to_eval.append(ti)
                ei_to_eval.append(ei)
        output = []
        if len(ti_to_eval) > 0:
            team_inputs = get_inputs(ti_to_eval)
            enemy_inputs = get_inputs(ei_to_eval)
            outputs = self.inference(team_inputs, enemy_inputs)
            output = gn.utils_np.graphs_tuple_to_data_dicts(outputs[-1])
        v = []
        p = []
        for c,h in zip(is_cached, state_hashes):
            if c:
                cv, cp = cache[h]
                v.append(cv)
                p.append(cp)
            else:
                dict = output.pop(0)
                v.append(dict["globals"][0])
                p.append(dict["nodes"])
                cache[h] = (v[-1], p[-1])
        return v,p

    # def predict(self, team_input_dicts, enemy_input_dicts, state_hashes):
    #
    #     team_inputs = get_inputs(team_input_dicts)
    #     enemy_inputs = get_inputs(enemy_input_dicts)
    #     outputs = self.model(team_inputs, enemy_inputs, self.config["test_mp_iterations"])
    #     output = gn.utils_np.graphs_tuple_to_data_dicts(outputs[-1])
    #     return [dict["globals"][0] for dict in output], [dict["nodes"] for dict in output]




class TrainableModel(InferenceModel):

    def __init__(self, config, team_input_example, enemy_input_example, target_example):
        super(TrainableModel, self).__init__(config, team_input_example, enemy_input_example)
        optimizer = snt.optimizers.Adam(config["learning_rate"])

        # Training.
        def update_step(team_inputs_tr, enemy_inputs_tr, targets_tr):
            with tf.GradientTape() as tape:
                outputs_tr = self.model(team_inputs_tr, enemy_inputs_tr, config["train_mp_iterations"])
                # Loss.
                ce_losses, mse_losses = make_loss(targets_tr, outputs_tr)
                ce_loss = tf.math.reduce_sum(ce_losses) / config["train_mp_iterations"]
                mse_loss = tf.math.reduce_sum(mse_losses) / config["train_mp_iterations"]
                regularizer = snt.regularizers.L2(config["l2_regularization"])
                train_loss = ce_loss + mse_loss + regularizer(self.model.trainable_variables)

                gradients = tape.gradient(train_loss, self.model.trainable_variables)
                optimizer.apply(gradients, self.model.trainable_variables)
                return outputs_tr, train_loss, ce_loss, mse_loss

        team_input_example = get_inputs([team_input_example])
        enemy_input_example = get_inputs([enemy_input_example])
        target_example = get_targets([target_example])

        # Get the input signature for that function by obtaining the specs
        input_signature = [
          gn.utils_tf.specs_from_graphs_tuple(team_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(enemy_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(target_example, dynamic_num_graphs=True)
        ]

        # Compile the update function using the input signature for speedy code.
        self.step_op = tf.function(update_step, input_signature=input_signature)


    def train(self, team_input_dicts, enemy_input_dicts, target_dicts):
        # for team,en,tar in zip(team_input_dicts, enemy_input_dicts, target_dicts):
        #     print(team,en,tar)
        _, loss, ce, mse = self.step_op(get_inputs(team_input_dicts), get_inputs(enemy_input_dicts), get_targets(target_dicts))
        return loss, ce, mse


    def test(self, team_input_dicts, enemy_input_dicts, target_dicts):
        team_inputs = get_inputs(team_input_dicts)
        enemy_inputs = get_inputs(enemy_input_dicts)
        targets = get_targets(target_dicts)
        outputs = self.model(team_inputs, enemy_inputs, self.config["test_mp_iterations"])
        _, loss = make_loss(targets, outputs)
        return loss
