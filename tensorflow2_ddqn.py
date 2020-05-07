import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import os
from replay_memory import ReplayBuffer
from utils import plot_learning_curve, make_env
#this is my own implementation of the dqn code given my mr phil using tensorflow 2
#tensorflow2 really is easy to code in. this code was developed using google colab
#requires a GPU as of tensorflow version 2.0.1 i think

class DQN(keras.Model):
    def __init__(self,lr,n_actions,name,input_dims,chkpt_dir):
        super(DQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        #by default conv2D uses format (NWHC): (batch, width,height,channels)
        #format ncwh requires a gpu at the moment that is why we are here at google colab
        
        self.conv1 = keras.layers.Conv2D(input_shape=(-1,*input_dims),
            filters=32,kernel_size=8,data_format='channels_first',strides=4,activation='relu')
        #need the channels first data format in order to use the shape(32,4,84,84) format NCWH
        # but now the problem becomes that i actually need a GPU, which is bad for now but I can live with it
        #testing in google colab is positive
        self.conv2 = keras.layers.Conv2D(filters=32,kernel_size=4,strides=2,data_format='channels_first',
            activation='relu')
        self.conv3 = keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,data_format='channels_first',
            activation='relu')

        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(units=512, activation='relu')
        self.dense2 = keras.layers.Dense(units=n_actions)

        #make prepareations in order to move operatiions to proper device. for local runtime

    def call(self, state):

        #print('input shape: ',state.shape)
        #exit()
        conv1 = self.conv1(state)
        #print('Conv1 output shape:', conv1.shape)
        conv2 = self.conv2(conv1)
        #print('Conv2 output shape:', conv2.shape)
        conv3 = self.conv3(conv2)
        #print('Conv3 output shape:',conv3.shape)
        flat = self.flat(conv3)
        #print('flat : ', flat.shape)
        dense1 = self.dense1(flat)
        #print('dense1 shape: ', dense1.shape)
        actions = self.dense2(dense1)
        #print('actions: ', actions.shape)
        return actions

    def save_checkpoint(self):
      pass
      print('Saving weights...')
      self.save(self.checkpoint_file)

    def load_checkpoint(self):
      pass
      print('loading checkpoint...')
      self.load_weights(self.checkpoint_file)


class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DQN(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DQN(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, observation):
        if tf.random.uniform([1]) > self.epsilon:

          actions = self.q_eval.call(tf.expand_dims(observation, axis=0))
     
          action = tf.argmax(actions, axis=1)

        else:
          action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state,action,reward,state_,done)

    def sample_memory(self):
        states,actions,rewards,new_states,dones = self.memory.sample_buffer(self.batch_size)

        return states, actions, rewards, new_states,dones

    def replace_target_network(self):
        if self.learn_step_counter & self.replace_target_cnt ==0:
            self.q_next = self.q_eval

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
        
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
    
        self.replace_target_network()
        states,actions,rewards,states_,dones = self.sample_memory()
        optimizer = keras.optimizers.RMSprop(learning_rate=self.lr)
        indices = tf.range(self.batch_size)

        with tf.GradientTape() as tape:

            q_pred = tf.gather_nd(self.q_eval.call(states), list(zip(indices, actions)))
            q_next = self.q_next.call(states_)
            q_eval = self.q_eval.call(states_)
            max_actions = tf.math.argmax(q_eval, axis=1)
            q_next_np = np.array(q_next)
            q_next_np[np.array(dones,dtype=np.bool)] = 0
            q_next = tf.convert_to_tensor(q_next_np, dtype=tf.int32)
            gather = tf.gather_nd(q_next, list(zip(indices, tf.cast(max_actions, dtype=tf.int32))))
            q_target = rewards + self.gamma * tf.cast(gather, dtype=tf.float32)

            loss = keras.losses.MSE(q_target, q_pred)
        gradient = tape.gradient(loss, self.q_eval.trainable_variables)
        optimizer.apply_gradients(zip(gradient, self.q_eval.trainable_variables))
        self.decrement_epsilon()
        self.learn_step_counter += 1

      
  

def main(num_games= 10,load_checkpoint=False, env_name='PongNoFrameskip-v4'):
    env  = make_env(env_name)
    best_score = -np.inf

    agent = DQNAgent(gamma=0.99, epsilon=1.0,lr=0.0001,input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=20000, eps_min=0.1, batch_size=32,replace=1000,
                     eps_dec=1e-5, chkpt_dir='models/',algo='DQNAgent', env_name=env_name)
    if load_checkpoint:
          agent.load_models()
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(num_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    n_steps = 0
    scores, eps_history,steps_array = [], [], []

    for i in range(num_games):
          done = False
          observation = env.reset()
          score = 0
          while not done:
                action = agent.choose_action(observation)
                observation_,reward,done,info = env.step(action)
                score += reward
                if not load_checkpoint:
                      agent.store_transition(observation, action, reward,observation_, int(done))
                      agent.learn()

                observation = observation_
          scores.append(score)
          steps_array.append(n_steps)
          avg_score = np.mean(scores[-100:])
          print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
          
          if avg_score > best_score:
            #if not load_checkpoint:
            #    agent.save_models()
              best_score = avg_score

          eps_history.append(agent.epsilon)
          if load_checkpoint and n_steps >= 18000:
              break

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

      

main(100)

  
      