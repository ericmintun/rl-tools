'''
Learns Atari games from OpenAI gym using a basic double deep learning
algorithm.
'''

import gym
import rl
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import gym_utils.wrappers as wrap

def main():
    #Initialize environment
    env_name = 'PongNoFrameskip-v4'
    obs_shape = (210, 160, 3)
    action_space = (6,)
    action_at_start='FIRE'
    action_repeat = 4
    max_rand_noop = 30
    env = wrap.EpisodeEndOnReward(
            wrap.ActionRepeat(
                wrap.RandomNoOpAtStart(
                    wrap.ActionAtStart(
                        gym.make(env_name), 
                        action_at_start),
                    max_rand_noop),
                action_repeat))

    use_gpu = torch.cuda.is_available()

    #Image processor initialization
    initial_color_pos = 'after'
    new_color_pos = 'before'
    crop_size = None
    offset = None
    grey_scale = True
    coarse_grain_factor = 3
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
    frame_step = 1
    frame_copy = True
    flatten_channels = True
    pre = rl.preprocessors.PixelPreprocessor(obs_shape,
                                             img,
                                             frame_number,
                                             frame_step,
                                             frame_copy,
                                             flatten_channels)

    #Replay memory initialization
    mem_size = 20000
    mem_init_size = 1000
    memory = rl.utils.agent.ReplayMemoryNumpy(mem_size,
                                              pre.output_shape)

    #Network initialization
    network_input_shape = pre.output_shape
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
    if use_gpu:
        network.cuda()

    #Postprocessor initialization
    post = rl.postprocessors.DiscreteQPostprocessor()

    #Loss initialization
    loss_layer = torch.nn.MSELoss()

    #Optimizer initialization
    #learning_rate = 1e-4
    #optim = torch.optim.SGD(network.parameters(), learning_rate)
    optim = torch.optim.Adam(network.parameters())

    #Actor initialization
    max_e = 1
    min_e = 0.02
    frames_to_min = int(1e5)
    e_slope = ( max_e - min_e ) / frames_to_min
    e_function = lambda n : max(max_e - n * e_slope, min_e)

    actor = rl.actors.DiscreteEpsilonGreedy(action_space[0], e_function)

    #Agent initialization
    discount = 0.99
    batch_size = 32
    update_snapshot = int(1e4)
    double_deep_q = False

    agent = rl.agents.DeepQ(pre, memory, network, post, loss_layer, optim, 
                            actor, discount, batch_size, update_snapshot, 
                            double_deep_q, use_gpu)

    #Loop initialization
    training_episodes = int(4e4)
    episodes_per_render = int(100)
    episodes_per_update = int(100)
    frames_per_update = int(1e4)

    total_frames = 0
    start_time = time.time()
    current_time = start_time
    episode_rewards = np.zeros(training_episodes)
    avg_decay = 0.99
    episode_rewards_avg = np.zeros(training_episodes)

    #Pyplot initialization
    #plt.ion()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.pause(0.05)
    #image_update = 10

    #Replay memory initialization
    old_obs = pre.process(env.reset())
    for i in range(mem_init_size):
        rand_action = np.random.choice(np.arange(action_space[0]))
        raw_obs, reward, done, info = env.step(rand_action)
        new_obs = pre.process(raw_obs)
        memory.add(old_obs, rand_action, reward, new_obs, done)
        if done:
            old_obs = pre.process(env.reset())
            pre.reset_episode()
        else:
            old_obs = new_obs
        
    #Main training loop
    for i in range(training_episodes):

        episode_reward = 0

        #Reset saved preprocessor states
        pre.reset_episode()

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
                env.render()
                #if total_frames % image_update == 0:
                #    processed = img.process(observation)
                #    ax.imshow(processed.reshape(processed.shape[1], processed.shape[2]), cmap='Greys')
                #    fig.canvas.draw()
                #    plt.pause(0.05)
            action = agent.train(observation, reward, terminated)
            total_frames += 1
            episode_reward += reward
            if total_frames % frames_per_update == 0:
                running_time = time.time() - current_time
                current_time = time.time()
                print("Completed frame number " + str(total_frames) + ". The last " \
                        + str(frames_per_update) + " were completed in " \
                        + "{:3.0f}".format(running_time) + " seconds.")

        #Update reward records
        episode_rewards[i] = episode_reward
        if i != 0:
            episode_rewards_avg[i] = episode_rewards_avg[i-1] * avg_decay \
                                     + episode_reward * (1- avg_decay)
        else:
            episode_rewards_avg[0] = episode_reward

        if (i+1) % episodes_per_update == 0:
            print("Episode " + str(i+1) + " completed. Net reward: "  \
                + str(episode_rewards[i]) + ". Running average: " \
                + "{:2.3f}".format(episode_rewards_avg[i]))


        #Update plots
        #plt.cla()
        #plt.plot(episode_rewards)
        #plt.plot(episode_rewards_avg)
        #plt.draw()

    #Keep plots around
    #plt.show()
    
if __name__ == "__main__":
    main()
