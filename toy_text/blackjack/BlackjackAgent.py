class BlackjackAgent:
    def __init__(self,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor: float = 0.95
                 ):
        self.q_values = dict(lambda: np.zeros(env.action_dpace.n))
