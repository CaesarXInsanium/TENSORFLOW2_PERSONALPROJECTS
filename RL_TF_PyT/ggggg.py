import gym
env = gym.make('Breakout-v0')
while True:
    env.reset()
    done = False
    if not done:
        env.render()
        _, reward,done, _ = env.step(env.action_space.sample())
