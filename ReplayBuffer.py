from collections import deque
import random

class Replay_Buffer(object):
    def __init__(self, buffer_size=10e6, batch_size=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=int(buffer_size))

    def __call__(self):
        return self.memory

    def store_transition(self, transition):
        self.memory.append(transition)

    def store_transitions(self, transitions):
        if len(self.memory)<self.buffer_size:
            self.memory.extend(transitions)
        else:
            self.memory.popleft()
            self.memory.extend(transitions)

    def get_batch(self, batch_size=None):
        b_s = batch_size or self.batch_size
        cur_men_size = len(self.memory)
        if cur_men_size < b_s:
            return random.sample(list(self.memory), cur_men_size)
        else:
            return random.sample(list(self.memory), b_s)

    def memory_state(self):
        return {"buffer_size": self.buffer_size,
                "current_size": len(self.memory),
                "full": len(self.memory)==self.buffer_size}

    def empty_transition(self):
        self.memory.clear()

if __name__ == '__main__':
    pass