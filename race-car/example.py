import pygame
import random
from src.game.core import initialize_game_state, game_loop
from solution.human_controller import HumanController
from solution.queue_controller import QueueController



if __name__ == '__main__':
    # seed_value = 1234
    # seed_value = 565318
    # seed_value = 8678959
    # seed_value = 3805586
    # seed_value = 437694
    # seed_value = 209238
    # seed_value = 4097591
    # seed_value = 7753498
    seed_value = random.randint(0, 9999999)
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True, controller=QueueController())
    # game_loop(verbose=True, controller=HumanController())
    print(seed_value)
    pygame.quit()