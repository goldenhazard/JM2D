import os
import numpy as np
import pickle

from d3il.agents.utils.sim_path import sim_framework_path

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories of the D3IL Avoiding demonstrations.

    Observation = (x_des, y_des, x, y) of the end effector, action = (dx_des, dy_des).
    The data is read from ``<d3il>/environments/dataset/data/avoiding/data`` (see README).

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if env == 'avoiding-d3il' or env == 'd3il-avoiding':
        data_directory = 'environments/dataset/data/avoiding/data'
        data_dir = sim_framework_path(data_directory)
        state_files = os.listdir(data_dir)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]

                input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

                vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
                valid_len = len(vel_state)

            episode_data = {
                'observations': input_state[:-1],
                'actions': vel_state,
                'rewards': np.zeros(valid_len),
                'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
            }

            yield episode_data
    else:
        raise NotImplementedError(env)
