import tensorflow as tf

class ReplayMem(object):
    def __init__(self, max_size, input_shape, n_actions, testing=False):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_mem = tf.zeros((self.mem_size, *input_shape), dtype=tf.float32)
        self.new_state_mem = tf.zeros((self.mem_size, *input_shape), dtype=tf.float32)
        self.action_mem = tf.zeros((self.mem_size), dtype=tf.int32)
        self.reward_mem = tf.zeros((self.mem_size), dtype=tf.float32)
        self.term_mem = tf.zeros((self.mem_size), dtype=tf.int16)
        if testing:
            self.state_mem = tf.random.uniform((self.mem_size, *input_shape), dtype=tf.float32)
            self.new_state_mem = tf.random.uniform((self.mem_size, *input_shape), dtype=tf.float32)
            self.action_mem = tf.zeros([self.mem_size], dtype=tf.int32)
            self.reward_mem = tf.random.uniform([self.mem_size], dtype=tf.float32)
            self.term_mem = tf.zeros([self.mem_size], dtype=tf.int16)

    def store_transition(self, state, action, reward, state_, done: bool):
        self.state_mem = tf.concat([self.state_mem[1:], [state]], axis=0)
        self.new_state_mem = tf.concat([self.new_state_mem[1:], [state_]], axis=0)
        self.action_mem = tf.concat([self.action_mem[1:], [action]], axis=0)
        self.reward_mem = tf.concat([self.reward_mem[1:], [reward]], axis=0)
        self.term_mem = tf.concat([self.term_mem[1:], [done]], axis=0)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        batch = tf.random.shuffle(tf.expand_dims(tf.range(batch_size), 1))

        states = tf.gather_nd(self.state_mem, batch)
        actions = tf.gather_nd(self.action_mem, batch)
        rewards = tf.gather_nd(self.reward_mem, batch)
        states_ = tf.gather_nd(self.new_state_mem, batch)
        terminal = tf.gather_nd(self.term_mem, batch)

        return states, actions, rewards, states_, terminal

