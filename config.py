config = {
    #model latent dimensions
    "enemy_reducer_edge_layers" : [50]*1,
    "enemy_reducer_node_layers" : [50]*1,
    "enemy_reducer_global_layers" : [50]*1,

    "encode_edge_layers" : [50]*1,
    "encode_node_layers" : [50]*1,
    "encode_global_layers" : [50]*1,

    "core_edge_layers" : [50]*3,
    "core_node_layers" : [50]*3,
    "core_global_layers" : [50]*3,

    #reinforcement
    "num_actions" : 5,

    "max_trajectory_length" : 1000,
    "buffer_size" : 10000,

    "tree_simulations" : 1,

    "random_moves_init" : 20,

    "root_dirichlet_alpha" : 0.06,
    "root_exploration_fraction" : 0.25,

    "pb_c_base" : 19652,
    "pb_c_init" : 4,

    "num_trajectories" : 24,

    #message passing
    "train_mp_iterations" : 5,
    "test_mp_iterations" : 5,

    #training
    "learning_rate" : 1e-3,
    "batch_size" : 4000,
    "epochs" : 1,
    "train_iterations" : 5,
    "l2_regularization" : 1e-4,
    "num_testing_scenes" : 50,

    #performance
    "threads" : 8,
    "cached_batch" : 1000 #number of mcts iterations with cached inference
}
