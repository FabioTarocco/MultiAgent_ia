"""DDQN agent script 

This manages the training phase of the off-policy DDQN.
"""

import random
from collections import deque
import time
import yaml
import numpy as np

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()
    
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(seed)

from utils.deepnetwork import DeepNetwork
from utils.memorybuffer import Buffer

class DDQN:
    """
    Class for the DQN agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)

        Returns:
            None
        """

        """
        self.env = env

        self.model = DeepNetwork.build(env, params['dnn'])
        self.model_tg = DeepNetwork.build(env, params['dnn'])
        self.model_tg.set_weights(self.model.get_weights())
        self.model_opt = Adam()

        self.buffer = Buffer(params['buffer']['size'])
        """
        self.env = env
        self.model=[]
        self.model_tg = []
        self.buffer = []
        self.model_opt = []
        
        for i in range (0,self.env.n - 1):
            self.model.append(DeepNetwork.build(env, params['dnn']))
            self.model_tg.append(DeepNetwork.build(env, params['dnn']))
            self.buffer.append(Buffer(params['buffer']['size']))
            self.model_tg[i].set_weights(self.model[i].get_weights())
            self.model_opt.append(Adam())
            #self.model_opt[i] = Adam()
        
        
    def get_action(self, state, eps):
        """Get the action to perform

        Args:
            state (list): agent current state
            eps (float): random action probability

        Returns:
            action (float): sampled actions to perform
        """

        """
        One hot encoding
        Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        Agente 1 : [0, 0, 0, 0, 1]
        Agente 2 : [0, 0, 0, 1, 0]
        ...
        Agente env.n: [0, 0, 0, 0, 1]
        il get_action deve tornare una roba simile [[0,0,0,0,1]
                                                    [0,0,0,1,0]
                                                    ...
                                                    [0,0,0,0,1]]
        Matrice nx5 00001
                    00010
                    ...
                    00001
        
        self.discrete_action_space = True
        if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        """
        actions = np.zeros((self.env.n, self.env.action_space[0].n),dtype='int')
        for i in range (self.env.n - 1):
            if np.random.uniform() <= eps:
                actions[i][np.random.randint(0, self.env.action_space[0].n)]=1
            else:
                q_values = self.model(np.array([state])).numpy()[0]    
                actions[i][np.argmax(q_values)] = 1
        return actions

    def update(self, gamma, batch_size):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C

        Returns:
            None
        """
        for i in range (self.env.n -1):
            batch_size = min(self.buffer[i].size, batch_size)
            states, actions, rewards, obs_states, dones = self.buffer[i].sample(batch_size)

        # The updates require shape (n° samples, len(metric))
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)

            self.fit(gamma, states, actions, rewards, obs_states, dones, self.model[i], self.model_tg[i], self.model_opt[i])

    def fit(self, gamma, states, actions, rewards, obs_states, dones, model, model_tg, model_opt ):
        """We want to minimizing mse of the temporal difference error given by Q(s,a|θ) and the target y = r + γ max_a' Q_tg(s', a'|θ). It addresses the non-stationary target of DQN

        Args:
            gamma (float): discount factor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """
        
        with tf.GradientTape() as tape:
            # Compute the target y = r + γ max_a' Q_tg(s', a'|θ), where a' is computed with model
            obs_qvalues_tg = model_tg(obs_states).numpy()
            
            obs_qvalues = model(obs_states)
            obs_actions = tf.math.argmax(obs_qvalues, axis=-1).numpy()
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(obs_actions)])

            max_obs_qvalues = tf.expand_dims(tf.gather_nd(obs_qvalues_tg, idxs), axis=-1)
            y = rewards + gamma * max_obs_qvalues * dones

            # Compute values Q(s,a|θ)
            qvalues = model(states)
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(actions)])
            qvalues = tf.expand_dims(tf.gather_nd(qvalues, idxs), axis=-1)

            # Compute the loss as mse of Q(s, a) - y
            td_errors = tf.math.subtract(qvalues, y)
            td_errors = 0.5 * tf.math.square(td_errors)
            loss = tf.math.reduce_mean(td_errors)

            # Compute the model gradient and update the network
            grad = tape.gradient(loss, self.model.trainable_variables)
            model_opt.apply_gradients(zip(grad,model.trainable_variables))

 
    @tf.function
    def polyak_update(self, weights, target_weights, tau):
        """Polyak update for the target networks

        Args:
            weights (list): network weights
            target_weights (list): target network weights
            tau (float): controls the update rate

        Returns:
            None
        """

        for (w, tw) in zip(weights, target_weights):
            tw.assign(w * tau + tw * (1 - tau))

    def train(self, tracker, n_episodes, verbose, params, hyperp):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., the critic's gamma)
            hyperp (dict): algorithmic specific values (e.g., tau)

        Returns:
            None
        """

        mean_good_reward = deque(maxlen=100)
        mean_adv_reward = deque(maxlen=100)

  
        eps, eps_min = params['eps'], params['eps_min']
        eps_decay = hyperp['eps_d'] 
        tau, use_polyak, tg_update = hyperp['tau'], hyperp['use_polyak'], hyperp['tg_update']

        for e in range(n_episodes):
            ep_good_reward,ep_adv_reward, steps = 0, 0, 0


            state = self.env.reset()
            print(state)
            badTH = 1000000
            for s in state:
                badTH = min(badTH, s.size)

            while steps < 250:
                action = self.get_action(state, eps)
                a = np.zeros(self.env.action_space[0].n)
                a[action] = 1
                obs_state, obs_reward, done, _ = self.env.step([a])

                #obs_state = obs_state[-1]
                #obs_reward = obs_reward[-1]
                #done = done[-1]
                #print(obs_state,obs_reward)

                for i in range (self.env.n - 1):
                    self.buffer[i].store(state[i], 
                        action[i], 
                        obs_reward[i], 
                        obs_state[i], 
                        1 - int(done[i])
                    )

                for i in range (self.env.n):
                    if obs_state[i].size>badTH:
                        ep_good_reward+=obs_reward[i]
                    else:
                        ep_adv_reward+=obs_reward[i]

                #ep_adv_reward.append(ep_adv_reward)
                #ep_good_reward.append(ep_single_good_reward)

                #ep_reward += obs_reward
                steps += 1

                state = obs_state
                
                if e > params['update_start']: 
                    self.update(
                        params['gamma'], 
                        params['buffer']['batch']
                    )
                    
                    for i in range (self.env.n - 1):            
                        if use_polyak:
                            # DDPG Polyak update improve stability over the periodical full copy
                            self.polyak_update(self.model[i].variables, self.model_tg[i].variables, tau)
                        elif steps % tg_update == 0:
                            self.model_tg[i].set_weights(self.model[i].get_weights())
                
                if done: break  

            eps = max(eps_min, eps * eps_decay)

            mean_good_reward.append(ep_good_reward)
            mean_adv_reward.append(ep_adv_reward)
            tracker.update([e, ep_good_reward, ep_adv_reward])
            

            if e % verbose == 0: 
                tracker.save_metrics()
                #tracker.save_model(self.model, e, mean_good_reward[len(mean_good_reward) - 1], mean_adv_reward[len(mean_adv_reward) - 1])

           #if mean_reward[len(mean_reward)-1] < -20 : tracker.save_model(self.model,e,mean_reward[len(mean_reward)-1])

            print(f'Ep: {e}, Ep_Rew: {ep_good_reward}, Ep_Adv_Rew: {ep_adv_reward}, Mean_Rew: {np.mean(mean_good_reward)}, Mean_Adv_Rew: {np.mean(mean_adv_reward)}')
       


   