import numpy as np
import cart_pole_tabular_agent.CartPoleTabularAgent as cP

random_state = np.random.RandomState(20)
cutoff_time = 200

agent = cP.CartPoleTabularAgent()
agent.display_video = True

seed_agent = True
if seed_agent:
    agent.set_random_state(random_state)

agent.set_cutoff_time(cutoff_time)

agent.init_agent()

agent.restart_agent_learning()
agent.run()
