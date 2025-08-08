from enum import StrEnum

SENSOR_OPTIONS = [
    (90, "front"),
    (135, "right_front"),
    (180, "right_side"),
    (225, "right_back"),
    (270, "back"),
    (315, "left_back"),
    (0, "left_side"),
    (45, "left_front"),
    (22.5, "left_side_front"),
    (67.5, "front_left_front"),
    (112.5, "front_right_front"),
    (157.5, "right_side_front"),
    (202.5, "right_side_back"),
    (247.5, "back_right_back"),
    (292.5, "back_left_back"),
    (337.5, "left_side_back"),
]

SENSOR_OPTIONS_GUARANTEED = [
    (90, "front"),
    (180, "right_side"),
    (270, "back"),
    (0, "left_side"),
]

SENSOR_MAX_DISTANCE = 1000



class Actions(StrEnum):
    ACCELERATE = "ACCELERATE"
    DECELERATE = "DECELERATE"
    STEER_LEFT = "STEER_LEFT"
    STEER_RIGHT = "STEER_RIGHT"
    NOTHING = "NOTHING"


class GameState:
    def __init__(self, api_url: str):
        self.ego = None
        self.cars = []
        self.car_bucket = []
        self.sensors = []
        self.road = None
        self.statistics = None
        self.sensors_enabled = True
        self.api_url = api_url
        self.crashed = False
        self.elapsed_game_time = 0
        self.distance = 0
        self.latest_action = "NOTHING"
        self.ticks = 0
        self.actions = []