import pygame
from time import sleep
from .game_state import GameState, SENSOR_OPTIONS
from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector
from solution.controller import Controller
from dtos import RaceCarPredictRequestDto
import json

# Define constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
CAR_COLORS = ['yellow', 'blue', 'red']
MAX_TICKS = 60 * 60  # 60 seconds @ 60 fps
MAX_MS = 60 * 1000600   # 60 seconds flat

# Define game state
STATE = None


def intersects(rect1, rect2):
    return rect1.colliderect(rect2)

# Game logic
def handle_action(action: str):
    if action == "ACCELERATE":
        STATE.ego.speed_up()
    elif action == "DECELERATE":
        STATE.ego.slow_down()
    elif action == "STEER_LEFT":
        STATE.ego.turn(-0.1)
    elif action == "STEER_RIGHT":
        STATE.ego.turn(0.1)
    else:
        pass

def update_cars():
    for car in STATE.cars:
        car.update(STATE.ego)


def remove_passed_cars():
    min_distance = -1000
    max_distance = SCREEN_WIDTH + 1000
    cars_to_keep = []
    cars_to_retire = []

    for car in STATE.cars:
        if car.x < min_distance or car.x > max_distance:
            cars_to_retire.append(car)
        else:
            cars_to_keep.append(car)

    for car in cars_to_retire:
        STATE.car_bucket.append(car)
        car.lane = None

    STATE.cars = cars_to_keep

def place_car():
    if len(STATE.cars) > LANE_COUNT:
        return

    speed_coeff_modifier = 5
    x_offset_behind = -0.5
    x_offset_in_front = 1.5

    open_lanes = [lane for lane in STATE.road.lanes if not any(c.lane == lane for c in STATE.cars if c != STATE.ego)]
    lane = random_choice(open_lanes)
    x_offset = random_choice([x_offset_behind, x_offset_in_front])
    horizontal_velocity_coefficient = random_number() * speed_coeff_modifier

    car = STATE.car_bucket.pop() if STATE.car_bucket else None
    if not car:
        return

    velocity_x = STATE.ego.velocity.x + horizontal_velocity_coefficient if x_offset == x_offset_behind else STATE.ego.velocity.x - horizontal_velocity_coefficient
    car.velocity = Vector(velocity_x, 0)
    STATE.cars.append(car)

    car_sprite = car.sprite
    car.x = (SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
    car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
    car.lane = lane


def get_action_json():
    """
    Get action depending on tick from the actions_log.json.
    Finds the action for the current STATE.ticks.
    """
    try:
        with open("actions_log.json", "r") as f:
            actions = json.load(f)
            for entry in actions:
                if entry.get("tick") == STATE.ticks:
                    return entry.get("action", "NOTHING")
            return "NOTHING"
    except FileNotFoundError:
        return "NOTHING"


def initialize_game_state( api_url: str, seed_value: str, sensor_removal = 0):
    seed(seed_value)
    global STATE
    STATE = GameState(api_url)

    # Create environment
    STATE.road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
    middle_lane = STATE.road.middle_lane()
    lane_height = STATE.road.get_lane_height()

    # Create ego car
    ego_velocity = Vector(10, 0)
    STATE.ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
    ego_sprite = STATE.ego.sprite
    STATE.ego.x = (SCREEN_WIDTH // 2) - (ego_sprite.get_width() // 2)
    STATE.ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - ego_sprite.get_height() / 2)
    sensor_options = SENSOR_OPTIONS.copy()

    for _ in range(sensor_removal): # Removes random sensors
        random_sensor = random_choice(sensor_options)
        sensor_options.remove(random_sensor)
    STATE.sensors = [
        Sensor(STATE.ego, angle, name, STATE)
        for angle, name in sensor_options
    ]

    # Create other cars and add to car bucket
    for i in range(0, LANE_COUNT - 1):
        car_colors = ["blue", "red"]
        color = random_choice(car_colors)
        car = Car(color, Vector(8, 0), target_height=int(lane_height * 0.8))
        STATE.car_bucket.append(car)

    STATE.cars = [STATE.ego]

    return STATE


def get_action(controller: Controller) -> str:
    if len(STATE.actions) == 0:
        STATE.actions = controller.get_action(
            RaceCarPredictRequestDto(
                did_crash=STATE.crashed,
                elapsed_ticks=STATE.ticks,
                distance=STATE.distance,
                velocity={"x": STATE.ego.velocity.x, "y": STATE.ego.velocity.y},
                sensors={ sensor.name: sensor.reading for sensor in STATE.sensors if sensor is not None}
            )
        ).actions

    if len(STATE.actions) > 0:
        return STATE.actions.pop(0)

    raise ValueError("Empty action list returned.")


def update_game(current_action: str, render: bool = False, screen: pygame.Surface = None):
    
    # handle_action(current_action)
    # STATE.distance += STATE.ego.velocity.x
    # update_cars()
    # remove_passed_cars()
    # place_car()
    # for sensor in STATE.sensors:
    #    sensor.update()

    global STATE
    clock = pygame.time.Clock()

    delta = clock.tick(60)  # Limit to 60 FPS
    STATE.elapsed_game_time += delta
    STATE.ticks += 1


    if STATE.crashed or STATE.ticks > MAX_TICKS or STATE.elapsed_game_time > MAX_MS:
        print(f"Game over: Crashed: {STATE.crashed}, Ticks: {STATE.ticks}, Elapsed time: {STATE.elapsed_game_time} ms, Distance: {STATE.distance}")
        # TODO: It should somehow signal the end of the round for the when training the RL agent
        # break

    handle_action(current_action)

    STATE.distance += STATE.ego.velocity.x
    update_cars()
    remove_passed_cars()
    place_car()

    # print("Current action:", action)
    # print("Currnet tick:", STATE.ticks)

    # Update sensors
    for sensor in STATE.sensors:
        sensor.update()

    STATE.latest_action = current_action
    
    # Handle collisions
    for car in STATE.cars:
        if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
            STATE.crashed = True
    
    # Check collision with walls
    for wall in STATE.road.walls:
        if intersects(STATE.ego.rect, wall.rect):
            STATE.crashed = True

    if render:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0, 0, 0))  # Clear the screen with black

        # Draw the road background
        screen.blit(STATE.road.surface, (0, 0))

        # Draw all walls
        for wall in STATE.road.walls:
            wall.draw(screen)

        # Draw all cars
        for car in STATE.cars:
            if car.sprite:
                screen.blit(car.sprite, (car.x, car.y))
                bounds = car.get_bounds()
                color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                pygame.draw.rect(screen, color, bounds, width=2)
            else:
                pygame.draw.rect(screen, (255, 255, 0) if car == STATE.ego else (0, 0, 255), car.rect)

        # Draw sensors if enabled
        if STATE.sensors_enabled:
            for sensor in STATE.sensors:
                sensor.draw(screen)
    
        # Draw controller visualization
        # controller.visualize(screen, STATE)

        pygame.display.flip()


    return STATE



# Main game loop
ACTION_LOG = []

def game_loop(controller: Controller, verbose: bool = True, log_actions: bool = True, log_path: str = "actions_log.json"):
    global STATE
    clock = pygame.time.Clock()
    screen = None
    if verbose:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Race Car Game")



    while True:
        # delta = clock.tick(60)  # Limit to 60 FPS
        delta = clock.tick(240)
        # delta = clock.tick(240 if STATE.ticks < 400 else 3)
        # delta = clock.tick(240 if STATE.ticks < 270 else 3)
        STATE.elapsed_game_time += delta
        STATE.ticks += 1


        if STATE.crashed or STATE.ticks > MAX_TICKS or STATE.elapsed_game_time > MAX_MS:
            print(f"Game over: Crashed: {STATE.crashed}, Ticks: {STATE.ticks}, Elapsed time: {STATE.elapsed_game_time} ms, Distance: {STATE.distance}")
            break

        action = get_action(controller)

        # Log the action with tick
        if log_actions:
            ACTION_LOG.append({"tick": STATE.ticks, "action": action})

        handle_action(action)

        STATE.distance += STATE.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()

        # print("Current action:", action)
        # print("Currnet tick:", STATE.ticks)

        # Update sensors
        for sensor in STATE.sensors:
            sensor.update()
        
        # Handle collisions
        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                STATE.crashed = True
        
        # Check collision with walls
        for wall in STATE.road.walls:
            if intersects(STATE.ego.rect, wall.rect):
                STATE.crashed = True

        # Render game (only if verbose)
        if verbose:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            screen.fill((0, 0, 0))  # Clear the screen with black

            # Draw the road background
            screen.blit(STATE.road.surface, (0, 0))

            # Draw all walls
            for wall in STATE.road.walls:
                wall.draw(screen)

            # Draw all cars
            for car in STATE.cars:
                if car.sprite:
                    screen.blit(car.sprite, (car.x, car.y))
                    bounds = car.get_bounds()
                    color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                    pygame.draw.rect(screen, color, bounds, width=2)
                else:
                    pygame.draw.rect(screen, (255, 255, 0) if car == STATE.ego else (0, 0, 255), car.rect)

            # Draw sensors if enabled
            if STATE.sensors_enabled:
                for sensor in STATE.sensors:
                    sensor.draw(screen)
        
            # Draw controller visualization
            controller.visualize(screen, STATE)

            pygame.display.flip()


        STATE.latest_action = action

    # # Save actions to file after game ends
    # import os
    # if log_actions:
    #     log_dir = os.path.dirname(log_path)
    #     if log_dir and not os.path.exists(log_dir):
    #         os.makedirs(log_dir, exist_ok=True)
    #     with open(log_path, "w") as f:
    #         json.dump(ACTION_LOG, f, indent=2)

# Initialization - not used
def init(api_url: str):
    global STATE
    STATE = GameState(api_url)
    print(f"Game initialized with API URL: {api_url}")


# Entry point
if __name__ == "__main__":
    seed_value = None
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)  # Replace with actual API URL
    game_loop(verbose=True)  # Change to verbose=False for headless mode
    pygame.quit()