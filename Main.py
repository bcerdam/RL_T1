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


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    EPSILON = [0, 0.01, 0.1]
    NUM_OF_ARMS = 10

    resultados = []
    for epsilon in EPSILON:
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):

            bandit = SimpleBanditEnv(seed=run_id, num_of_arms=NUM_OF_ARMS)
            agent = SimpleBanditAgent(NUM_OF_ARMS)
            best_action = bandit.best_action

            for _ in range(NUM_OF_STEPS):
                action = agent.get_action(epsilon)
                reward = bandit.step(action)
                agent.learn(action, reward)

                is_best_action = action == best_action
                results.add_result(reward, is_best_action)
            results.save_current_run()

        avgr, oap = show_results(results)
        resultados.append([avgr, oap])



    ### CODIGO PLOTTEO PREGUNTA a) ###

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # vivid_cmap = plt.get_cmap('tab10')
    # colors = vivid_cmap(np.linspace(0, 1, len(EPSILON)))
    #
    # for i, (epsilon_value, data) in enumerate(zip(EPSILON, resultados)):
    #     average_rewards = data[0]
    #     optimal_actions = data[1]
    #     current_color = colors[i]
    #     label_text = f'ε = {epsilon_value}'
    #     axs[0].plot(average_rewards, label=label_text, color=current_color, alpha=0.75)
    #     axs[1].plot(optimal_actions, label=label_text, color=current_color, alpha=0.75)
    #
    # axs[0].set_title('Average Reward')
    # axs[0].set_ylabel('Average Reward')
    # axs[0].legend()
    # axs[0].grid(True, linestyle='--', alpha=0.6)
    # axs[1].set_title('% Optimal Action')
    # axs[1].set_ylabel('% Optimal Action')
    # axs[1].set_xlabel('Steps')
    # axs[1].set_ylim(0, 1)
    # axs[1].legend()
    # axs[1].grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # # plt.savefig('resultados/pregunta_a.jpeg', dpi=500)
    # plt.show()

    ### CODIGO PLOTTEO PREGUNTA a) ###