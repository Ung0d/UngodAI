config = {
    #model latent dimensions
    "enemy_reducer_edge_layers" : [100]*1,
    "enemy_reducer_node_layers" : [100]*1,
    "enemy_reducer_global_layers" : [100]*1,

    "encode_edge_layers" : [100]*2,
    "encode_node_layers" : [100]*2,
    "encode_global_layers" : [100]*2,

    "core_edge_layers" : [100]*2,
    "core_node_layers" : [100]*2,
    "core_global_layers" : [100]*2,

    #reinforcement
    "num_actions" : 5,

    "max_trajectory_length" : 1000,
    "buffer_size" : 10000,

    "tree_simulations" : 400,

    "random_moves_init" : 20,

    "root_dirichlet_alpha" : 0.06,
    "root_exploration_fraction" : 0.25,

    "pb_c_base" : 19652,
    "pb_c_init" : 4,


    "num_trajectories" : 100,
    #message passing
    "train_mp_iterations" : 10,
    "test_mp_iterations" : 10,

    #training
    "learning_rate" : 1e-3,
    "batch_size" : 4000,
    "epochs" : 1000,
    "train_iterations" : 40,
    "l2_regularization" : 1e-4,

    #performance
    "threads" : 4,
    "cached_batch" : 1000 #number of mcts iterations with cached inference
}
