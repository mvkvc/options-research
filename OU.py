import numpy as np
import random
class OU_Process(object):
    def __init__(self, action_dim,type, theta=0.15, mu=0, sigma=0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.type=type
        self.current_x = None
        self.init_process()

    def init_process(self):
        self.current_x = np.ones(self.action_dim) * self.mu

    def update_process(self,x_input):
        # dx = self.theta * (self.mu - x_input) + self.sigma * np.random.normal(loc=0.5)
        # dx = self.sigma * np.random.randn(self.action_dim)
        # dx = 0
        x_next =np.random.normal(x_input,1)
        if self.type == 'C':
            # dx = self.theta * (self.mu - x_input) + random.uniform(0.15, 1)
            x_next=np.clip(x_next,0,1)
        if self.type == 'P':
            # dx = self.theta * (self.mu - x_input) + random.uniform(-1, -0.15)
            x_next = np.clip(x_next,-1,0)
        # x_next = x_input + dx
        return x_next

    def return_noise(self,x_input):
        return self.update_process(x_input)



if __name__ == "__main__":
    pass
