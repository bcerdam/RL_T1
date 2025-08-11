import matplotlib.pyplot as plt
import numpy as np
from SimpleBanditEnv import SimpleBanditEnv
from BanditResults import BanditResults
from agents.SimpleBanditAgent import SimpleBanditAgent


def show_results(bandit_results: type(BanditResults)) -> tuple:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")
    return average_rewards, optimal_action_percentage

'''
alpha = -1.0 -> Step size no constante, osea = 1/N
'''
def bandit_simulation(epsilon: float, num_of_arms: int, num_of_steps: int, num_of_runs: int, alpha: float, initial_action_value_estimate: int) -> tuple:

    results = BanditResults()
    for run_id in range(num_of_runs):

        bandit = SimpleBanditEnv(seed=run_id, num_of_arms=num_of_arms)
        agent = SimpleBanditAgent(num_of_arms, initial_action_value_estimate)
        best_action = bandit.best_action

        for _ in range(num_of_steps):
            action = agent.get_action(epsilon)
            reward = bandit.step(action)
            agent.learn(action, reward, alpha)

            is_best_action = (action == best_action)
            results.add_result(reward, is_best_action)
        results.save_current_run()

    average_rewards, optimal_action_percentage = show_results(results)
    return average_rewards, optimal_action_percentage


def plot_full_comparison(all_results, parameter_name='epsilon'):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    vivid_cmap = plt.get_cmap('tab10')
    colors = vivid_cmap(np.linspace(0, 1, len(all_results)))

    for i, result_data in enumerate(all_results):
        label_text = f"{parameter_name} = {result_data['parameter']}"
        axs[0].plot(result_data['rewards'], label=label_text, color=colors[i])
        axs[1].plot(result_data['actions'], label=label_text, color=colors[i])

    axs[0].set_title('Average Reward Comparison')
    axs[0].set_ylabel('Average Reward')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_title('% Optimal Action Comparison')
    axs[1].set_ylabel('% Optimal Action')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylim(0, 1)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.savefig('resultados/pregunta_a.jpeg', dpi=500)
    plt.show()

def plot_optimal_action_comparison(all_results, parameter_name='epsilon'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    vivid_cmap = plt.get_cmap('tab10')
    colors = vivid_cmap(np.linspace(0, 1, len(all_results)))

    for i, result_data in enumerate(all_results):
        label_text = f" {result_data['parameter']}"
        ax.plot(result_data['actions'], label=label_text, color=colors[i])

    ax.set_title('% Optimal Action Comparison')
    ax.set_ylabel('% Optimal Action')
    ax.set_xlabel('Steps')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.savefig('resultados/pregunta_c.jpeg', dpi=500)
    plt.show()

if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    NUM_OF_ARMS = 10

    ''' Pregunta a '''

    # EPSILON_VALUES = [0.0, 0.01, 0.1]
    #
    # results = []
    # for epsilon in EPSILON_VALUES:
    #     avg_rewards, optimal_actions = bandit_simulation(epsilon, NUM_OF_ARMS, NUM_OF_STEPS, NUM_OF_RUNS, alpha=-1.0, initial_action_value_estimate=0)
    #     results.append({'parameter': epsilon, 'actions': optimal_actions, 'rewards': avg_rewards})
    #
    # plot_full_comparison(results, parameter_name='epsilon')
    
    ''' Pregunta a '''
    
    
    ''' Pregunta c '''

    # ALPHA = 0.1
    # params = [[0.0, 5], [0.1, 0]]
    # results = []
    #
    # for param in params:
    #     avg_rewards, optimal_actions = bandit_simulation(param[0], NUM_OF_ARMS, NUM_OF_STEPS, NUM_OF_RUNS, ALPHA, param[1])
    #     results.append({'parameter': f'epsilon={param[0]}, Q_1={param[1]}', 'actions': optimal_actions})
    #
    # plot_optimal_action_comparison(results, parameter_name='')
    
    ''' Pregunta c '''
