'''
Various agents.

Currently non-functioning and half pseudo-code.
'''

import utils.agent

class DoubleDeepQ:

    def __init__(self, network, preprocessor, trainer, discount=0.99, 
                 memory_size=1000, input_frames=4, batch_size=50, 
                 update_snapshot=10000):
        self.network = network
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.memory_size = memory_size
        self.input_frames = input_frames
        self.batch_size = batch_size
        self.update_snapshot = update_snapshot
        self.discount = discount

        self.last_action = None
        self.last_obs = None


    def train(self, state):
        
        processed = self.preprocessor.process(state)
        transition = {'initial' : self.last_obs, 'action' : self.last_action, 'final' : processed, 'reward' : state['reward'], 'done' : state['done']}

        if self.memory.is_full:
            batch = self.memory.get_batch(self.batch_size-1)
            batch.append(transition)
            self.memory.add(transition)
        else:
            batch = transition

        y_pred = self.postprocess.q(self.network.forward(batch['initial'],fixed=False))
        q_target = self.postprocess.q(self.network.forward(batch['final'],fixed=True))
        y_target = batch['reward'] + discount * q_target * batch['done']

        self.trainer.train(y_pred, y_target)

        action = self.postprocess.action(






        
