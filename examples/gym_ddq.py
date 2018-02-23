'''
Learns Atari games from OpenAI gym using a basic double deep learning
algorithm.
'''

import gym
import rl
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    env_name = 'Pong-v0'
    obs_shape = (210, 160, 3)
    action_space = (6,)
    env = gym.make(env_name)

    use_gpu = torch.cuda.is_available()

    #Image processor initialization
    initial_color_pos = 'after'
    new_color_pos = 'before'
    crop_size = None
    offset = None
    grey_scale = True
    coarse_grain_factor = 5
    old_max_value = 256
    new_max_value = 1

    img = rl.utils.image.ImageProcessor(initial_color_pos,
                                        new_color_pos,
                                        crop_size,
                                        offset,
                                        grey_scale,
                                        coarse_grain_factor,
                                        old_max_value,
                                        new_max_value)
    
    #Preprocessor initialization
    frame_number = 4
    frame_step = 2
    frame_copy = True
    flatten_channels = True
    pre = rl.preprocessors.PixelPreprocessor(img,
                                             frame_number,
                                             frame_step,
                                             frame_copy,
                                             flatten_channels)

    #Network initialization
    network_input_shape = pre.output_shape(obs_shape)
    network_output_shape = action_space
    kernel_size = 5
    conv1_features = 32
    conv2_features = 64
    lin1_features = 1024
    pool_kernel_size = 2
    dropout_p = 0.5

    network = rl.rl_cnn.RL_CNN(network_input_shape,
                               network_output_shape,
                               kernel_size,
                               conv1_features,
                               conv2_features,
                               lin1_features,
                               pool_kernel_size,
                               dropout_p)

    #Postprocessor initialization
    post = rl.postprocessors.DiscreteQPostprocessor()

    #Loss initialization
    loss_layer = torch.nn.MSELoss()

    #Optimizer initialization
    learning_rate = 1e-5
    optim = torch.optim.SGD(network.parameters(), learning_rate)

    #Actor initialization
    max_e = 1
    min_e = 0.05
    frames_to_min = 1e4
    e_slope = ( max_e - min_e ) / frames_to_min
    e_function = lambda n : max(max_e - n * e_slope, min_e)

    actor = rl.actors.DiscreteEpsilonGreedy(action_space[0], e_function)

    #Agent initialization
    discount = 1
    memory_size = 1000
    batch_size = 50
    update_snapshot = int(1e3)
    double_deep_q = True

    agent = rl.agents.DeepQ(pre, network, post, loss_layer, optim, actor,
                            discount, memory_size, batch_size,
                            update_snapshot, double_deep_q, use_gpu)

    #Loop initialization
    training_episodes = int(200)
    episodes_per_render = int(10)
    console_update = int(1000)

    total_frames = 0
    episode_rewards = np.zeros(training_episodes)
    avg_decay = 0.9
    episode_rewards_avg = np.zeros(training_episodes)
    for i in range(training_episodes):

        episode_reward = 0

        #Initialize new episode and take first step
        observation = env.reset()
        reward = 0
        terminated = False
        info = {}

        action = agent.train(observation, reward, terminated)
        total_frames += 1
        episode_reward += reward

        #Train!
        while not terminated:
            observation, reward, terminated, info = env.step(action)
            if i % episodes_per_render == 0:
                #env.render()
                pass
            action = agent.train(observation, reward, terminated)
            total_frames += 1
            episode_reward += reward

        #Update reward records
        episode_rewards[i] = episode_reward
        if i != 0:
            episode_rewards_avg[i] = episode_rewards_avg[i-1] * avg_decay \
                                     + episode_reward * (1- avg_decay)
        else:
            episode_rewards_avg[0] = episode_reward
        print("Episode " + str(i) + " completed. Net reward: " + str(episode_rewards[i])
                + ". Running average: " + str(episode_rewards_avg[i]))

        #Update plots
        plt.cla()
        plt.plot(episode_rewards)
        plt.plot(episode_rewards_avg)
        plt.draw()

    #Keep plots around
    plt.show()
    
if __name__ == "__main__":
    main()
