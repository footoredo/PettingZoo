import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_targets=1):
        world = World()
        # add agents
        world.dim_p = 2
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_targets)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            shade = i / (len(world.landmarks) - 1)
            landmark.color = np.array([0.75, 0.75, 0.75]) - shade * 0.25
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.steps = 0

    def get_info(self, agent, world):
        return [np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) for landmark in world.landmarks]

    def reward(self, agent, world):
        dist2 = min([np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) for landmark in world.landmarks])
        r = 0.
        # for landmark in world.landmarks:
        #     if np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) < agent.size + landmark.size + 0.01:
        #         r += 5.
        return -dist2 + r

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([np.array([world.steps])] + [agent.state.p_vel] + entity_pos)

    def get_input_structure(self, agent, world):
        dim_p = world.dim_p
        input_structure = list()
        input_structure.append(("self", dim_p + 1))
        for _ in world.landmarks:
            input_structure.append(("landmarks", dim_p))
        return input_structure
