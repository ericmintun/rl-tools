'''
Various agents.

Currently non-functioning and half pseudo-code.
'''

from utils.agent import Transition, ReplayMemory

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

        self.memory = ReplayMemory(self.memory_size)

        self.last_action = None
        self.last_obs = None


    def train(self, observation, reward, terminated, info=None):

        #Preprocess the observation
        obs_processed = self.preprocessor.process(observation)

        #If first frame of episode, take an action without training.
        if self.last_obs == None:
            #Put action taking here
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

        #Predict rewards for performed actions
        init_obs_nn_input = self.preprocessor.batchify(batch.init_obs)
        init_obs_nn_output = self.network.forward(init_obs_nn_input)
        q_pred = self.postprocessor.estimated_reward(init_obs_nn_output,
                                                      batch.actions)

        #Determine best rewards for states after performed actions
        #The use of snapshot parameters implements the double in double deep q.
        final_obs_nn_input = self.preprocessor.batchify(batch.final_obs)
        final_obs_nn_output = self.network.forward(init_obs_nn_output,
                                                    use_snapshot=True)
        actions, q_final = self.postprocessor.best_action(final_obs_nn_output,
                                                           output_q=True)

        #Pseudo-code below this point.

        #Target for state is reward recieved + next state estimated reward.
        #If state is terminal, no further estimated reward is included.
        q_target = batch.reward + discount * q_final * (not batch.done)

        #Calculate loss and backpropagate
        loss = self.loss_function(q_pred, q_target, self.network.parameters)
        loss.backward()

        #Update parameters
        self.optim.step(self.network.parameters)

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
        best_action = actions[-1] #Last element in batch is the current state.
        action = determine_action(best_action) #Actual action on policy?
        self.last_action = action
        self.last_obs = obs_processed
        return action
        




        
