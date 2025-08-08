from pygame import Surface

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from src.game.game_state import GameState, SENSOR_OPTIONS, SENSOR_OPTIONS_GUARANTEED, SENSOR_MAX_DISTANCE, Actions



class Controller:
    def get_action(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto:
        raise NotImplementedError("This method should be overridden by subclasses")

    def visualize(self, screen: Surface, state: GameState) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")