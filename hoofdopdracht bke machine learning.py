from bke import MLAgent, is_winner, opponent, train, validate, RandomAgent, plot_validation

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
    
my_agent = MyAgent(alpha=0.8, epsilon=0.2)

my_agent = MyAgent()

train(my_agent, 3000)

my_agent.learning = False

validation_agent = RandomAgent()
 
validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=100)
 
plot_validation(validation_result)
