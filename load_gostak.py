import textworld

env = textworld.start("tw_games/gostak.z5")

state = env.reset()
print(state.feedback)

state, score, done = env.step("look")
print(state.feedback)

env.close()

