from grid2op.Action import PlayableAction
import numpy as np


def init_all_actions(env):
    all_actions = [env.action_space({})]

    for node_id in range(env.observation_space.dim_topo):
        if node_id in env.observation_space.gen_pos_topo_vect:
            gen_id = np.argwhere(env.observation_space.gen_pos_topo_vect == node_id)[0][0]
            for bus_id in range(1, 3):
                action = env.action_space({'set_bus': {'generators_id': [(gen_id, bus_id)]}})
                all_actions.append(action)
        if node_id in env.observation_space.load_pos_topo_vect:
            load_id = np.argwhere(env.observation_space.load_pos_topo_vect == node_id)[0][0]
            for bus_id in range(1, 3):
                action = env.action_space({'set_bus': {'loads_id': [(load_id, bus_id)]}})
                all_actions.append(action)
        if node_id in env.observation_space.line_or_pos_topo_vect:
            line_or_id = np.argwhere(env.observation_space.line_or_pos_topo_vect == node_id)[0][0]
            for bus_id in range(1, 3):
                action = env.action_space({'set_bus': {'lines_or_id': [(line_or_id, bus_id)]}})
                all_actions.append(action)
        if node_id in env.observation_space.line_ex_pos_topo_vect:
            line_ex_id = np.argwhere(env.observation_space.line_ex_pos_topo_vect == node_id)[0][0]
            for bus_id in range(1, 3):
                action = env.action_space({'set_bus': {'lines_ex_id': [(line_ex_id, bus_id)]}})
                all_actions.append(action)

    return all_actions


class TopologyBusSetAction(PlayableAction):
    """
    This type of :class:`PlayableAction` implements the modifications
    of the grid with "change" topological actions.
    It accepts the key words: "change_line_status" and "change_bus".
    Nothing else is supported and any attempt to use something else
    will have no impact.
    """
    authorized_keys = {
        "set_bus"
    }

    attr_list_vect = [
        "_set_topo_vect"
    ]
    attr_list_set = set(attr_list_vect)

    def __init__(self):
        super().__init__()
