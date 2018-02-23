'''
Various agents.
'''

from rl.utils.agent import Transition, ReplayMemory, batchify
import torch
from torch.autograd import Variable
import numpy as np

class DeepQ:
    '''
    An agent for deep Q learning.  There may be a way to do this much
    better.
    
    Major limitations:
    No regularization possible in loss function
    Action choice can depend on the best action only.

    Initialization arguments:
    preprocessor (Preprocessor) : class in charge of translating raw
      observations into a form the neural network can take as input.
    network (RLModule) : the neural network to train.  Must have a
      forward method that can take a use_snapshot flag
    postprocessor (Postprocessor) : class in charge of turning the 
      output of the neural network into definitive actions and estimated
      rewards.
    loss_layer (torch.nn._Loss) : any torch loss layer that with a 
      forward method that takes predictions and targets
    optim (torch.nn.Optimizer) : any torch optimizer
    actor (Action) : class that actually chooses an action to take. Must
      have an 'act' method that takes the best estimated action as input.
    discount (float) : amount to discount future rewards
    memory_size (int) : size of replay memory used to store previous
      states.
    batch_size (int) : minibatch size per training step
    update_snapshot (int) : the number of training steps to wait before
      updating the fixed network weights
    double_deep_q (bool) : If true, performs double deep q learning by
      using the fixed network to calculate target rewards.

    Methods:
    train(observation, reward, terminated, info) : trains the network on
      the given observation and reward.  Returns the action the network
      chooses to take, which will be None if terminated=True.  Info is
      unused.  Updates the agent's internal memory with its newest
      action, so this action should be applied to the environment before
      train is called again.

    best_action(observation) : returns the networks preferred action
      given the observation.  Does not advance the agent's internal
      memory about it's current state and should not be used in
      conjunction with train.
    '''

    def __init__(self, preprocessor, network, postprocessor, loss_layer, optim, 
                 actor, discount=0.99, memory_size=1000, batch_size=50, 
                 update_snapshot=10000, double_deep_q=False,use_gpu=False):
        self.preprocessor = preprocessor
        self.network = network
        self.postprocessor = postprocessor
        self.loss_layer = loss_layer
        self.optim = optim
        self.actor = actor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_snapshot = update_snapshot
        self.discount = discount
        self.ddq = double_deep_q

        self.memory = ReplayMemory(self.memory_size)
        self.steps_performed = 0

        self.last_action = None
        self.last_obs = None

        if use_gpu:
            self.torch_long = torch.LongTensor
            self.torch_float = torch.FloatTensor
        else:
            self.torch_long = torch.cuda.LongTensor
            self.torch_float = torch.cuda.FloatTensor


    def train(self, observation, reward, terminated, info=None):

        #Preprocess the observation
        obs_processed = self.preprocessor.process(observation)

        #If first frame of episode, take an action without training.
        if self.last_obs == None:
            best_action = self.best_action(observation)
            action = self.actor.act(best_action)
            self.last_action = action
            self.last_obs = obs_processed
            return action

        #Save transition to replay memory and get a new batch to train on
        transition = Transition(self.last_obs, self.last_action, 
                        reward, obs_processed, terminated)

        if self.memory.is_full:
            batch = self.memory.get_batch(self.batch_size-1)
            batch.append(transition) #Assure current transition is included
            self.memory.add(transition)
        else:
            batch = transition #If memory isn't full use only current transition

        #Organize numerical output
        batch_actions = batchify(batch.action)
        batch_rewards = batchify(batch.reward)
        batch_done = batchify(batch.done).astype(int)

        #Predict rewards for performed actions
        init_obs_nn_input = Variable(torch.from_numpy(
                self.preprocessor.batchify(batch.init_obs)).type(self.torch_float))
        init_obs_nn_output = self.network.forward(init_obs_nn_input)
        q_pred = self.postprocessor.estimated_reward(init_obs_nn_output, 
                                                      batch_actions)


        #Determine best rewards for states after performed actions
        #The use of snapshot parameters implements the double in double deep q.
        final_obs_nn_input = Variable(torch.from_numpy(
                self.preprocessor.batchify(batch.final_obs)).type(self.torch_float))
        final_obs_nn_output = self.network.forward(final_obs_nn_input,
                                                    use_snapshot=self.ddq)
        actions, q_final = self.postprocessor.best_action(final_obs_nn_output,
                                                           output_q=True)

        #Target for state is reward recieved + next state estimated reward.
        #If state is terminal, no further estimated reward is included.
        rewards = Variable(torch.from_numpy(batch_rewards).type(self.torch_float))
        torch_done = torch.from_numpy(batch_done).type(self.torch_float)
        done_mask = Variable(torch.ones(torch_done.size()).type(self.torch_float)
                              -torch_done)
        
        q_target = rewards + self.discount * q_final * done_mask

        #Calculate loss and backpropagate
        loss = self.loss_layer(q_pred, q_target)
        loss.backward()

        #Update parameters
        self.optim.step()

        #Update snapshots if it's time
        self.steps_performed += 1
        if self.steps_performed % self.update_snapshot == 0:
            self.network.update_snapshot()

        #If we've reached the final state, clean up and return no action.
        if terminated == True:
            self.last_action = None
            self.last_obs = None
            return None

        #Otherwise determine action to take and save current state.

        #If performing double deep q learning, already have relevant action.
        if self.ddq:
            best_action = actions[-1] #Last element in batch is the current state.
            action = self.actor.act(best_action)
            self.last_action = action
            self.last_obs = obs_processed
            return action

        #Otherwise need to calculate with snapshot
        best_action = self.best_action(observation)
        action = self.actor.act(best_action)
        self.last_action = action
        self.last_obs = obs_processed
        return action
    

    def best_action(self, observation):
        obs_processed = self.preprocessor.process(observation)
        obs_nn_input = Variable(torch.from_numpy(
                self.preprocessor.batchify(obs_processed)).type(self.torch_float))
        obs_nn_output = self.network.forward(obs_nn_input, use_snapshot=True)
        best_action = self.postprocessor.best_action(obs_nn_output)
        return best_action
        




        
