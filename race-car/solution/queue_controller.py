from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from math import inf, pi, radians, sin, cos, tan, floor, ceil, sqrt, log
sgn = lambda x: (x > 0) - (x < 0)

import pygame
from loguru import logger

from solution.controller import Surface, Controller, GameState, RaceCarPredictRequestDto, RaceCarPredictResponseDto, SENSOR_OPTIONS, SENSOR_OPTIONS_GUARANTEED, SENSOR_MAX_DISTANCE, Actions
from solution.human_controller import HumanController


type ActionHandler = callable[[RaceCarPredictRequestDto], RaceCarPredictResponseDto | None]

@dataclass()
class CarBounds:
    @dataclass()
    class Velocity:
        x: float
        tick_observed: int
    
    left: float = 0.0
    right: float = 0.0
    known: bool = False
    # Velocity and tick observed
    velocity: Velocity | None = None
    


SENSOR_NAME_ANGLE = {
    # NOTE: Convert from pygame angle madness to typical radians
    name: radians(-angle + 90)
    for angle, name in SENSOR_OPTIONS
}

PIXEL_EPSILON = 0.6



class QueueController(Controller):
    def __init__(
        self,
        max_sequence_length: int = 10,
        lane_count: int = 5,
        car_size: tuple[int, int] = (360, 179),

        # TODO: Edit these
        car_guess_max_width: int = 360*3,
        lane_hold_distance: int = 800,
        speed_hysteresis: float = 0.8,
        turn_hysteresis: float = 0.05,
        wait_stability_ticks: int = 60 * 6,
        speed_max_relative: float = 5.0, # NOTE: If set to negative then will slow down when doing the global reachable lane check for a switch and blind speed up
        # NOTE: Adding safety margin
        despawn_bounds: tuple[float, float] = (-1780 + 100, 1420 - 100),
        
        max_safe_speed_safety_margin: float = 0.0 # 5.0 + 0.5
    ) -> None:
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.lane_count = lane_count
        self.car_size = car_size
        self.car_guess_max_width = car_guess_max_width
        self.ticks = 0


        self.ego_accel_turn = 0.1
        self.ego_accel_x = 0.1
        self.car_accel_x = 0.1

        self.lane_hold_distance = lane_hold_distance
        self.speed_hysteresis = speed_hysteresis
        self.turn_hysteresis = turn_hysteresis
        self.wait_stability_ticks = wait_stability_ticks
        self.speed_max_relative = speed_max_relative
        self.despawn_bounds = despawn_bounds


        self.lanes_reachable_prev = [(False, False) for _ in range(self.lane_count)]
        


        # NOTE: Maximum speed before braking becomes unviable
        # NOTE: Safety margin is a heuristic for how much fast the car in front is going and is going to decelerate
        self.max_safe_speed = sqrt(2000 * self.ego_accel_x) - max_safe_speed_safety_margin
        
        
        
        self.wait_stability_start: int | None = None
        
        # TODO: Is a lane change better?
        self.lane_switch_tick_prev = 0
        
        
        
        
        
        
        
        self.map_height = 1120.0
        self.lane_height = self.map_height / self.lane_count
        self.lane_centers: list[float] = [0.0] * self.lane_count
        self.ego_vel_x = 10.0
        self.ego_vel_y = 0.0
        self.ego_pos_y = 0.0
        self.ego_lane_idx = 2
        self.sensor_max: dict[str, float] = {}
        # NOTE: Assuming batched actions are not long enough to cause uncertainty about car position.
        #  This may happen in practice, but it is mitigated by always guessingg the car is in front.
        self.car_pos_x: list[CarBounds | None] = [None] * self.lane_count
        self.car_pos_y_near: list[float] = [0.0] * self.lane_count
        self.car_pos_y_far: list[float] = [0.0] * self.lane_count

        self.handler_queue: deque[ActionHandler] = deque()
        self.handler_queue.extend([
            # Initial calibration
            # NOTE: Assuming these stay constant
            # self._handler_calibrate_turn_right,
            # self._handler_calibrate_turn_left,
            # self._handler_calibrate_forward,
            # self._handler_calibrate_backward,
            # NOTE: Must take an action to get sensor data
            self._handler_accelerate,
            self._handler_calibrate_map,

            # Default driving
            # *[self._handler_steer_left] * 30,
            # self._handler_tailgate,
            # self._handler_automatic_control
            # accelerate till top speed before unable to brake
            # *[self._handler_steer_left]*40,
            # self._create_handler_lane_switch(0, None),
            # self._create_handler_lane_switch(2, None),
            # self._create_handler_lane_switch(0, None),
            # self._create_handler_lane_switch(4, None),
            # self._create_handler_lane_switch(1, None),
            # self._create_handler_lane_switch(2, None),
            # self._create_handler_lane_switch(1, None),
            # self._debug_handler_human,
            # *[self._handler_lane_keep] * 500,
            self._handler_planner
        ])

        self.request_prev: RaceCarPredictRequestDto | None = None
        self.response_prev: RaceCarPredictResponseDto | None = None



    # ===== Calibration handlers =====
    def _handler_calibrate_map(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        self.map_height = float(request.sensors["left_side"]) + float(request.sensors["right_side"])

        for i in range(self.lane_count):
            # NOTE: First lane is bottom lane
            lane_center = (i + 1/2) * self.lane_height - self.map_height / 2
            self.lane_centers[i] = lane_center

            down, up = lane_center - self.car_size[1] / 2, lane_center + self.car_size[1] / 2

            if self.ego_pos_y < lane_center:
                self.car_pos_y_far[i] = up
                self.car_pos_y_near[i] = down
            else:
                self.car_pos_y_far[i] = down
                self.car_pos_y_near[i] = up


        return None


    # ===== Action handlers =====
    def _handler_accelerate(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE])

    def _handler_decelerate(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        return RaceCarPredictResponseDto(actions=[Actions.DECELERATE])

    def _handler_steer_left(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        return RaceCarPredictResponseDto(actions=[Actions.STEER_LEFT])

    def _handler_steer_right(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        return RaceCarPredictResponseDto(actions=[Actions.STEER_RIGHT])



    # ===== Debug handlers =====
    def _debug_handler_human(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        self.handler_queue.appendleft(self._debug_handler_human)
        print(self.ego_vel_x, self._calculate_speed_post_accel(self.ego_lane_idx, False), self.ego_vel_y)

        if not hasattr(self, "human_controller"):
            self.human_controller = HumanController()

        return self.human_controller.get_action(request)



    # ===== Automatic control handler =====
    def _handler_lane_keep(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        # Repeat the lane keep handler
        pos_hysteresis = self.lane_hold_distance * 0.1
        
        # Center the car in the lane via PD control
        control_output = 0.3*(self.ego_pos_y - self.lane_centers[self.ego_lane_idx]) + 3.0 * self.ego_vel_y
        
        if control_output > 1.0:
            return RaceCarPredictResponseDto(actions=[Actions.STEER_RIGHT])
        elif control_output < -1.0:
            return RaceCarPredictResponseDto(actions=[Actions.STEER_LEFT])
        
        # If below max safe speed and no car ahead, accelerate to safe speed
        car_bounds = self.car_pos_x[self.ego_lane_idx]
        
        # # If accelerate but car in front, ignore and handle it here instead
        # if len(self.handler_queue) > 0 and self.handler_queue[0] == self._handler_accelerate and car_bounds.left > self.car_size[0] / 2 + PIXEL_EPSILON:
        #     print("popped")
        #     self.handler_queue.popleft()

        # if self.ego_vel_x < self.max_safe_speed - self.speed_hysteresis and (not car_bounds or car_bounds.right < 0.0):
        #     return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE])
        # # Else if car on lane, keep distance
        # elif car_bounds:
        if car_bounds:
            # NOTE: Using last observed velocity instead of estimating worst case
            car_vel = car_bounds.velocity.x if car_bounds.velocity else 0

            # If car is in front and too close, decelerate
            if car_bounds.left > 0.0 and car_bounds.left + 5*car_vel - self.lane_hold_distance < pos_hysteresis:
                return RaceCarPredictResponseDto(actions=[Actions.DECELERATE])
            # if car_bounds.left > 0.0:
            #     self.handler_queue.appendleft(self._create_handler_lane_switch(self.ego_lane_idx+1, None))
            # Else if car is in front and too far, accelerate
            elif car_bounds.left > 0.0 and car_bounds.left + 5*car_vel - self.lane_hold_distance > pos_hysteresis:
                return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE])
            # # Else if car is behind and too close, accelerate
            # elif car_bounds.right < 0.0 and car_bounds.right + car_vel + self.lane_hold_distance > 0.0:
            #     return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE])
            # Else if car is behind, accelerate
            elif car_bounds.right < 0.0:
                return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE])

            # Match speed
            if abs(car_vel) < self.speed_hysteresis:
                return RaceCarPredictResponseDto(actions=[Actions.NOTHING] * self.max_sequence_length)
            else:
                return RaceCarPredictResponseDto(actions=[Actions.ACCELERATE if car_vel > 0.0 else Actions.DECELERATE] * max(1, min(self.max_sequence_length, int(abs(car_vel) // self.ego_accel_x) + 1)))
        # # Else no car in front, but above max safe speed, decelerate
        # elif self.ego_vel_x > self.max_safe_speed + self.speed_hysteresis:
        #     return RaceCarPredictResponseDto(actions=[Actions.DECELERATE])
        # Else maintain velocity
        else:
            return RaceCarPredictResponseDto(actions=[Actions.NOTHING])


    
    
    def _create_handler_lane_switch(self, lane_idx_target: int, speed_target: float | None = None) -> ActionHandler:
        def _handler(request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
            # If no lane switch is needed, pass back control
            if abs(self.ego_pos_y - self.lane_centers[lane_idx_target]) < PIXEL_EPSILON and abs(self.ego_vel_y) < self.turn_hysteresis:
                return None

            # If not at target speed, accelerate
            _speed_target = self.ego_vel_x if speed_target is None else speed_target
            speed_diff = _speed_target - self.ego_vel_x
            if abs(speed_diff) > self.speed_hysteresis:
                accel_ticks = max(1, min(self.max_sequence_length, int(round(abs(speed_diff) / self.ego_accel_x))))
                action = Actions.ACCELERATE if speed_diff > 0 else Actions.DECELERATE

                self.handler_queue.appendleft(_handler)
                return RaceCarPredictResponseDto(actions=[action] * accel_ticks)


            # NOTE: A direct lane switch is when the below code is used to change lanes.
            #  I.e. the lane switch only goes in one direction.

            # Calculate acceleration and deceleration ticks based on the distance to cover
            pos_diff = self.lane_centers[lane_idx_target] - self.ego_pos_y
            ticks_to_accel, ticks_to_decel = self._calculate_position_change_ticks_direct(pos_diff)

            # Create action response
            action_accel = Actions.STEER_LEFT if pos_diff > 0 else Actions.STEER_RIGHT
            action_decel = Actions.STEER_RIGHT if pos_diff > 0 else Actions.STEER_LEFT
            actions = [action_accel]*ticks_to_accel + [action_decel]*ticks_to_decel


            print(f"turning {ticks_to_accel}, {ticks_to_decel}")

            # If more actions, add handler to queue again
            if len(actions) > self.max_sequence_length:
                self.handler_queue.appendleft(_handler)
            # Else if no more actions, pass on control
            elif len(actions) == 0:
                return None


            # Return actions up to the maximum sequence length
            return RaceCarPredictResponseDto(actions=actions[:self.max_sequence_length])


        # Return specalized lane switch handler
        return _handler
    
    
    #  it is the relative that is important once you get them to despawn, it's all relative.
    #  So just need to figure out how to get past the first wave of cars.
    #  Then this process can be repeated whenever the speed is within a certain range
    # TODO: Accelerate up in speed, switch lanes instead of braking
    # TODO: FIgure out how much speed up is possible when passed a car, go into its lane, then accelerate, it despawns, spawns with our new speed +-, 
    #  when should we stop accelerating to avoid hitting it if it spawns in front
    #  Below is relative to screen center (ego pos x)
    #  Spawn location in the back: 1600*(-0,5)-180 - 1600/2 = -1780
    #  Spawn location in the front: 1600*(1,5)-180 - 1600/2 = 1420
    
    # It places down cars one at a time, no cars in the start, so in the start, the speed ranges are known

        
    # def _handler_brake(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
    #     car_bounds = self.car_pos_x[self.ego_lane_idx]
    #     if car_bounds is None or car_bounds.velocity is None or car_bounds.right < 0.0:
    #         return


    #     brake_distance = (car_bounds.velocity.x)**2 / (2 * self.ego_accel_x)
    #     if abs(brake_distance - (-self.car_size[0] / 2 + car_bounds.left)) < 200:
    #         self.handler_queue.appendleft(self._handler_lane_keep)
    #         return RaceCarPredictResponseDto(actions=[Actions.DECELERATE])

    
    # TODO: If manage to get in front of a car... 
    #  it's time to speed the fuck up while it's still in view,
    #  and continue doing it for 1000 estimated distance units past the screen view,
    #  because only then will the car despawn behind, and therefore risk spawning
    #  in front of the ego car.
    #  Passing cars can maybe be detected via side sensors relatively easily,
    #  now that we have the expected sensor max distances.
   
    
    # TODO: Figure out planner to weave between cars.





    # ===== Accident avoidance handlers =====
    # NOTE: These should realistically never happen
    
    # TODO: Emergency speed up or brake if a car is too close
    # TODO: Maybe evasive maneuvers too
    # TODO: Don't drive into walls. 


    # TODO: Ensure elapsed ticks is total elapsed ticks since the start of the game. Else all those calculations are wrong.



    # ===== Helpers =====
    def _calculate_position_change_ticks_direct(self, position_difference: float) -> tuple[int, int]:
            #  Equations:
            #   0 = y_vel+a*acc - a*dec
            #   pos_diff = acc*y_vel + a*acc*(acc-1) // 2 + dec*(y_vel+a*acc) - a*dec*(dec-1) // 2
            #   
            #   Solution:
            #   dec = y_vel/a+acc
            #   
            #   pos_diff = acc*y_vel + a*acc*(acc-1) // 2 + (y_vel/a+acc)*(y_vel+a*acc) - a*(y_vel/a+acc)*((y_vel/a+acc)-1) // 2
            #   Solve for acc
            v = self.ego_vel_y
            d = position_difference
            a = sgn(position_difference)*self.ego_accel_turn
            a = sgn(self.ego_vel_y) if position_difference == 0 else a
            # If no acceleration, then already at destination
            if a == 0:
                return 0, 0

            r = 4*a*d - 2*a*v + 2*v**2
            r = max(0.0, r)

            ticks_to_accel = floor((-v - sqrt(r)/2) / a)
            ticks_to_accel_alt = floor((-v + sqrt(r)/2) / a)
            ticks_to_accel = ticks_to_accel_alt if ticks_to_accel < 0 and ticks_to_accel_alt > ticks_to_accel else ticks_to_accel
            ticks_to_decel = ceil(v/a + ticks_to_accel)



            # # Fix off by ones caused by rounding
            # turn_final = abs(self.ego_vel_y) + self.ego_accel_turn*(ticks_to_accel-ticks_to_decel)
            # if position_difference > 0:
            #     if abs(turn_final - self.ego_accel_turn) < self.turn_hysteresis:
            #         ticks_to_decel += 1
            #     elif abs(turn_final + self.ego_accel_turn) < self.turn_hysteresis:
            #         ticks_to_accel += 1
            # elif position_difference < 0:
            #     if abs(turn_final - self.ego_accel_turn) < self.turn_hysteresis:
            #         ticks_to_accel += 1
            #     elif abs(turn_final + self.ego_accel_turn) < self.turn_hysteresis:
            #         ticks_to_decel += 1


            # TODO: Sometimes getting negative numbers. This is a mitigation.
            ticks_to_accel = max(0, ticks_to_accel)
            ticks_to_decel = max(0, ticks_to_decel)

            return ticks_to_accel, ticks_to_decel



    # NOTE: This method returns the GLOBAL speed
    def _calculate_speed_post_accel(self, lane_idx: int, pre_speed_match: bool) -> float:
        # NOTE: Assuming the lane is safe, lane change will be direct, successful, and constant car velocity
        # If no car in lane, assume it is safe to accelerate
        car_bounds = self.car_pos_x[lane_idx]
        if car_bounds is None:
            return self.ego_vel_x + self.speed_max_relative

        # If unknown car velocity, assume same as ego velocity
        car_vel = car_bounds.velocity.x if car_bounds.velocity is not None else 0.0

        # If car is in front, use car's velocity
        if car_bounds.left > 0.0:
            return self.ego_vel_x + car_vel*0.80
        # Else if car is directly adjacent, use ego velocity
        elif (-self.car_size[0]/2 <= car_bounds.left < self.car_size[0]/2) or (-self.car_size[0]/2 <= car_bounds.right < self.car_size[0]/2):
            return self.ego_vel_x
        # Else, car must be behind, use acceleration velocity
        else:
            left = car_bounds.left
            ego_vel = self.ego_vel_x

            # If speed matching before lane switch, account for speed matching
            if pre_speed_match:
                speed_match_ticks = abs(car_vel - self.ego_vel_x) // self.ego_accel_x
                left += car_vel*speed_match_ticks + sgn(self.ego_vel_x-car_vel)*self.ego_accel_x*(speed_match_ticks**2)/2
                ego_vel += car_vel - self.ego_vel_x
                car_vel = 0

            # Account for movement while switching lanes
            lane_diff = self.lane_centers[lane_idx] - self.ego_pos_y
            lane_switch_ticks = sum(self._calculate_position_change_ticks_direct(lane_diff))
            left += car_vel*lane_switch_ticks

            # TODO: Remove this hack once the speed estimate is more accurate
            if left < self.despawn_bounds[0]:
                left += 0.50*(car_bounds.right-left)
            
            # HACK: Fixes car that should be gone issues
            if left < self.despawn_bounds[0]:
                return self.ego_vel_x + self.speed_max_relative
            # TODO: Check if the acceleration amount is correct

            # Account for acceleration till car despawns
            # Equations:
            #  self.despawn_bounds[0] = car_vel*t - self.ego_accel_x*(t**2)/2 + left
            #  Solve for t
            r = max(0.0, 2*self.ego_accel_x*(left-self.despawn_bounds[0]) + car_vel**2)
            despawn_ticks = (car_vel + sqrt(r)) / self.ego_accel_x

            # TODO: And then finally add (or maybe substract something) the maximum relative speed
            return ego_vel + self.ego_accel_x*despawn_ticks + self.speed_max_relative









    def _find_lanes_reachable(self) -> list[tuple[bool, bool]]:
        lanes_reachable = [(False, False) for _ in range(self.lane_count)]


        # TODO: Not fully accurate, something is off



        switch_ticks = [
            self._calculate_position_change_ticks_direct(self.lane_centers[lane_idx] - self.ego_pos_y)
            for lane_idx in range(self.lane_count)
        ]
        switch_ticks_total = [max(1, sum(ticks)) for ticks in switch_ticks]

        car_bounds = deepcopy(self.car_pos_x)
        ego_pos_ys = [self.ego_pos_y for _ in range(self.lane_count)]
        ego_vel_ys = [self.ego_vel_y for _ in range(self.lane_count)]

        # Simulate lane switching for each target lane
        for t in range(1, max(switch_ticks_total) + 1):
            # Update car bounds based on velocity
            for car_bound in car_bounds:
                # NOTE: Assuming constant car velocity and zero car velocity if unknown
                if car_bound is None or car_bound.velocity is None:
                    continue

                # HACK: Adding fudge to make car bounds bigger
                car_bound.left += car_bound.velocity.x# - self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                car_bound.right += car_bound.velocity.x# + self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                # car_bound.left += car_bound.velocity.x - self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                # car_bound.right += car_bound.velocity.x + self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                # car_bound.left += car_bound.velocity.x - self.car_accel_x*max(1, log(abs(car_bound.velocity.x)*15+1e-6))*10
                # car_bound.right += car_bound.velocity.x + self.car_accel_x*max(1, log(abs(car_bound.velocity.x)*15+1e-6))*10

            # Update each target lane simulation
            for lane_idx in range(self.lane_count):
                if t > switch_ticks_total[lane_idx]:
                    continue

                # Update ego position and velocity
                turn_dir = sgn(self.lane_centers[lane_idx] - self.ego_pos_y)
                if t <= switch_ticks[lane_idx][0]:
                    ego_vel_ys[lane_idx] += turn_dir*self.ego_accel_turn
                elif t > switch_ticks[lane_idx][0]:
                    ego_vel_ys[lane_idx] -= turn_dir*self.ego_accel_turn

                ego_pos_ys[lane_idx] += ego_vel_ys[lane_idx]
                
                
                # Colision check with cars
                for i, car_bound in enumerate(car_bounds):
                    # NOTE: Assuming if no cars, then lane is clear
                    if car_bound is None:
                        continue
                    
                    # If car is in lane, check for collision
                    # HACK: Increasing safety margins
                    # x_margin = min(50, max(0, log((abs(car_bound.velocity.x) if car_bound.velocity else 0.0) + 1e-6)*20))
                    x_margin = 0.0
                    if car_bound.velocity and abs(car_bound.velocity.x) > 6 and min(abs(self.car_pos_x[i].left), abs(self.car_pos_x[i].right)) < 50:
                        # x_margin = 60
                        x_margin = 100
                    if abs(i - self.ego_lane_idx) > 1:
                        x_margin = 100
                    # x_margin = min(40, 0.95*abs(car_bound.velocity.x if car_bound.velocity else 0.0)**2)
                    y_margin = 20
                    x_collision = (
                        ((car_bound.right > -self.car_size[0] / 2 - PIXEL_EPSILON - x_margin) and
                        (car_bound.left < self.car_size[0] / 2 + PIXEL_EPSILON + x_margin)) \
                        or \
                        ((car_bound.left < self.car_size[0] / 2 + PIXEL_EPSILON + x_margin) and
                        (car_bound.right > -self.car_size[0] / 2 - PIXEL_EPSILON - x_margin))
                    )

                    car_top = max(self.car_pos_y_near[i], self.car_pos_y_far[i])
                    car_bottom = min(self.car_pos_y_near[i], self.car_pos_y_far[i])
                    y_collision = (
                        (car_bottom - PIXEL_EPSILON - y_margin <= ego_pos_ys[lane_idx] - self.car_size[1] / 2 <= car_top + PIXEL_EPSILON + y_margin) \
                        or \
                        (car_bottom - PIXEL_EPSILON - y_margin <= ego_pos_ys[lane_idx] + self.car_size[1] / 2 <= car_top + PIXEL_EPSILON + y_margin) \
                        or \
                        # NOTE: This is to handle an edge case where ego matches up exactly with the car top and bottom
                        (car_bottom - PIXEL_EPSILON - y_margin <= ego_pos_ys[lane_idx] <= car_top + PIXEL_EPSILON + y_margin)
                    )

                    # If collision or known but no velocity, set not reachable
                    if (x_collision and y_collision) or (car_bound.known and car_bound.velocity is None):
                        switch_ticks_total[lane_idx] = -1
                        break


                # If at destination, set reachable
                if t == switch_ticks_total[lane_idx]:
                    car_bound = car_bounds[lane_idx]
                    if car_bound is None or car_bound.velocity is None:
                        lanes_reachable[lane_idx] = (True, False)
                        switch_ticks_total[lane_idx] = -1
                        continue
                
                    speed_match_post_ticks = ceil(abs(car_bound.velocity.x - ego_vel_ys[lane_idx]) / self.ego_accel_x)
                    for i in range(speed_match_post_ticks):
                        # HACK: Adding fudge to make car bounds bigger
                        car_bound.left += car_bound.velocity.x# - self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                        car_bound.right += car_bound.velocity.x# + self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                        # car_bound.left += car_bound.velocity.x - self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                        # car_bound.right += car_bound.velocity.x + self.car_accel_x*t#*max(1, log(abs(car_bound.velocity.x)+1e-6))
                        # car_bound.left += car_bound.velocity.x - self.car_accel_x*max(1, log(abs(car_bound.velocity.x)+1e-6))*25
                        # car_bound.right += car_bound.velocity.x + self.car_accel_x*max(1, log(abs(car_bound.velocity.x)+1e-6))*25

                    x_collision = (
                        ((car_bound.right > -self.car_size[0] / 2 - PIXEL_EPSILON) and
                        (car_bound.left < self.car_size[0] / 2 + PIXEL_EPSILON)) \
                        or \
                        ((car_bound.left < self.car_size[0] / 2 + PIXEL_EPSILON) and
                        (car_bound.right > -self.car_size[0] / 2 - PIXEL_EPSILON))
                    )

                    if not x_collision:
                        # If no collision, set reachable with speed match
                        lanes_reachable[lane_idx] = (True, False)
                        switch_ticks_total[lane_idx] = -1
                
                # TODO: If reached with speed match, set reachable with speed match and no more exploring this lane
                #  Remember to update the collision stop searching condition
                
        # Handle lanes above
        
        


        # TODO: Also check whether to pre speed match


        return lanes_reachable












    # ===== Planning handlers =====
    def _handler_planner(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto | None:
        # # TODO: Temporary

        if self.ticks % 60 == 0:
            print(self.ego_vel_x)

        self.handler_queue.appendleft(self._handler_planner)

        lanes_reachable = self._find_lanes_reachable()

        # Check adjacent lanes for potential switches
        cars_adjacent = [
            (lane_idx, self.car_pos_x[lane_idx])
            for lane_idx in range(self.lane_count) if lanes_reachable[lane_idx][0] and abs(lane_idx - self.ego_lane_idx) == 1
        ]

        car_behind_speed = self._calculate_speed_post_accel(self.ego_lane_idx, lanes_reachable[self.ego_lane_idx][1])
        lane_idx_best = self.ego_lane_idx
        speed_best = car_behind_speed if self.car_pos_x[self.ego_lane_idx] else self.ego_vel_x
        for lane_idx, car_bound in cars_adjacent:
            if car_bound is None:
                continue

            # If car in front is faster than car behind, switch lanes
            if (speed := self._calculate_speed_post_accel(lane_idx, lanes_reachable[lane_idx][1])) > speed_best + self.speed_hysteresis:
                lane_idx_best = lane_idx
                speed_best = speed

        # If switching lanes
        # HACK: The reachability seems to be inconsistent, this smoothes it out slightly
        lanes_reachable_agreement = all(
            lanes_reachable[lane_idx][0] == self.lanes_reachable_prev[lane_idx][0] and
            lanes_reachable[lane_idx][1] == self.lanes_reachable_prev[lane_idx][1]
            for lane_idx in range(self.lane_count)
        )
        if lane_idx_best != self.ego_lane_idx and lanes_reachable_agreement:
            print("a")
            speed_factor = 1.0
            if self.car_pos_x[lane_idx_best] and self.car_pos_x[lane_idx_best].left > +self.car_size[0] / 2 + PIXEL_EPSILON:
                speed_factor = 0.90
            
            # If car in front, speed up to match
            if lanes_reachable[lane_idx_best][1]:
                self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, speed_best*speed_factor))
            else:
                self.handler_queue.extendleft([self._handler_accelerate] * max(1, int((speed_best*speed_factor - self.ego_vel_x) // self.ego_accel_x)))
                self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, None))

            self.wait_stability_start = None

            return None


        # if car is behind
        if self.car_pos_x[self.ego_lane_idx] and self.car_pos_x[self.ego_lane_idx].right < -self.car_size[0] / 2 - PIXEL_EPSILON:
            print("b")
            self.handler_queue.extendleft([self._handler_accelerate] * max(1, int((speed_best - self.ego_vel_x) // self.ego_accel_x)))
            self.wait_stability_start = None
            
            return None



        # If done waiting
        if self.wait_stability_start is not None and request.elapsed_ticks - self.wait_stability_start > self.wait_stability_ticks:
            print("impatient " * 4)
            # Check all lanes for potential switches
            car_bounds = [
                (lane_idx, self.car_pos_x[lane_idx])
                for lane_idx in range(self.lane_count) if lanes_reachable[lane_idx][0]
            ]

            lane_idx_best = self.ego_lane_idx
            speed_best = self.ego_vel_x

            for lane_idx, car_bound in car_bounds:
                if car_bound is None:
                    continue

                if (speed := self._calculate_speed_post_accel(lane_idx, lanes_reachable[lane_idx][1])) > speed_best + self.speed_hysteresis:
                    lane_idx_best = lane_idx
                    speed_best = speed

            # If switching lanes
            if lane_idx_best != self.ego_lane_idx and lanes_reachable_agreement:
                print("c")
                if lanes_reachable[lane_idx_best][1]:
                    self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, speed_best))
                else:
                    self.handler_queue.extendleft([self._handler_accelerate] * max(1, int((speed_best - self.ego_vel_x) // self.ego_accel_x)))
                    self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, None))

                self.wait_stability_start = None

                return None


            # Else if no cars to use, switch towards middle
            lane_idx_best = self.ego_lane_idx
            for lane_idx in range(self.lane_count):
                if lanes_reachable[lane_idx][0] and abs(lane_idx - 2) <= abs(lane_idx_best - 2) and abs(lane_idx - self.ego_lane_idx) < abs(lane_idx_best - self.ego_lane_idx):
                    lane_idx_best = lane_idx

            if lane_idx_best != self.ego_lane_idx:
                print("d")
                self.handler_queue.extendleft([self._handler_accelerate] * max(1, int(self.max_safe_speed // self.ego_accel_x)))
                self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, None))
                self.wait_stability_start = None

                return None
            # Else speed up in current lane
            if not self.car_pos_x[self.ego_lane_idx]:
                print("e")
                self.handler_queue.extendleft([self._handler_accelerate] * max(1, int(self.max_safe_speed / 2 // self.ego_accel_x)))
                self.wait_stability_start = None

            # lanes_reachable_agreement


        # # car_bounds = [
        # #     ((self.ego_lane_idx + 1, (self.car_pos_x[self.ego_lane_idx + 1] if self.ego_lane_idx + 1 < self.lane_count and lanes_reachable[self.ego_lane_idx + 1][0] else None))),
        # #     ((self.ego_lane_idx - 1, (self.car_pos_x[self.ego_lane_idx - 1] if self.ego_lane_idx - 1 >= 0 and lanes_reachable[self.ego_lane_idx - 1][0] else None)))
        # # ]
        # car_bounds = [
        #     (lane_idx, self.car_pos_x[lane_idx])
        #     for lane_idx in range(self.lane_count) if lanes_reachable[lane_idx][0]# and abs(lane_idx - self.ego_lane_idx) == 1
        # ]


        # lane_idx_best = self.ego_lane_idx
        # speed_best = self.ego_vel_x
        # for lane_idx, car_bound in car_bounds:
        #     if car_bound is None:
        #         continue

        #     if (speed := self._calculate_speed_post_accel(lane_idx, False)) > speed_best:
        #         lane_idx_best = lane_idx
        #         speed_best = speed

        # lanes_reachable_agreement = all(
        #     lanes_reachable[lane_idx][0] == self.lanes_reachable_prev[lane_idx][0] and
        #     lanes_reachable[lane_idx][1] == self.lanes_reachable_prev[lane_idx][1]
        #     for lane_idx in range(self.lane_count)
        # )
        # if lane_idx_best != self.ego_lane_idx and lanes_reachable_agreement:
        #     self.handler_queue.extendleft([self._handler_accelerate] * int((speed_best - self.ego_vel_x) // self.ego_accel_x / 2))
        #     self.handler_queue.appendleft(self._create_handler_lane_switch(lane_idx_best, None))



        # # if lane_idx_best != self.ego_lane_idx:
        # #     self.handler_queue.extendleft([self._handler_accelerate] * 25)

        # # self.handler_queue.appendleft(self._handler_lane_switch)
        # self.handler_queue.appendleft(self._handler_lane_keep)
        
        if self.wait_stability_start is None:
            self.wait_stability_start = request.elapsed_ticks
        
        
        self.handler_queue.appendleft(self._handler_lane_keep)
        
        self.lanes_reachable_prev = lanes_reachable
        
        # TODO: Planner must agree with itself for two ticks
        
        
        # TODO: maybe a lane is reachable... but then we get onto the lane and bang straight into the back of a car
        
        # TODO: Our lane switch code seems to overshoot sometimes
        
        
        return None
    

        lanes_reachable: list[int] = []
        


        if self.wait_stability_start is None:
            self.wait_stability_start = request.elapsed_ticks
        
        
        
        
        """
        when making decisions wrt speed, account for speed hysteresis
        when "waiting", we are waiting for cars to match our speed within some band through either respawn or through lack of observation of them for a while

        if ((adjacent lane car from behind after acceleration is faster than car behind after acceleration) or (adjacent lane car in front is faster than car behind after acceleration)) and can switch:
            switch lanes with speed match (before or after)
            reset waiting to none


        if car behind nearby (potentially no longer known but not yet despawned):
            accelerate till it despawns and then accelerate/decelerate to a speed that is safe
                we know exactly where it will spawn behind or in front
                reset waiting to none


        if no longer waiting:
            if (car behind in ANY lane or faster car in front in ANY lane is faster than car behind after acceleration) and can swittch:
                switch lanes with speed match (before or after)
                reset waiting to none
            elif there is an open reachable lane:
                switch lanes to an open lane with a bias towards the middle lane
            elif in open reachable lane closest to middle:
                accelerate up to next safe speed
                reset waiting to none
            else: # TODO: Later: Handle blocked lane situation where most or all lanes are blocked by cars
                slow down and switch one lane towards the middle


        if waiting is none:
            track tick for start of waiting


        if car in front:
            maintain speed and lane and some distance
        else:
            maintain speed and lane


        """
        
        # TODO: Dodging a car by switching lanes is quite good


        # TODO: The lane keep must auto-center
        
        # TODO: Figure out the math for how long/far to accelerate
        
        # TODO: Figure out car speed propagation more accurately
        
        
        # TODO: Multiple (lane switch methods) depending upon circumstances. 
        #  E.g. match speed turn, or turn match speed, ?or a mix?
        
        # TODO: Make target positions at edge lanes closer to the middle lane

        # TODO: Parallel testing of multiple runs to get point (distance) distribution


        return None
    


    # ===== Permanent handlers =====    
    def _update_ticks(self, request: RaceCarPredictRequestDto) -> None:
        self.ticks = request.elapsed_ticks
    
    def _update_ego_vel(self, request: RaceCarPredictRequestDto) -> None:
        self.ego_vel_x = float(request.velocity["x"])
        self.ego_vel_y = float(request.velocity["y"])


    def _update_ego_pos_y_wall(self, request: RaceCarPredictRequestDto) -> None:
        top = request.sensors["left_side"]
        bottom = request.sensors["right_side"]

        top = SENSOR_MAX_DISTANCE if top is None else float(top)
        bottom = SENSOR_MAX_DISTANCE if bottom is None else float(bottom)

        # If is accurate wall-based measurement, set zero position to be the middle
        if abs(top + bottom - self.map_height) < PIXEL_EPSILON:
            self.ego_pos_y = bottom - self.map_height / 2


    def _estimate_ego_pos_y_dead_reckoning(self) -> None:
        # If first update, no reason to do dead reckoning
        if self.response_prev is None:
            return

        vel_vertical = float(self.request_prev.velocity["y"])

        for action in self.response_prev.actions:
            if action == Actions.STEER_LEFT:
                vel_vertical += self.ego_accel_turn
            elif action == Actions.STEER_RIGHT:
                vel_vertical -= self.ego_accel_turn

            self.ego_pos_y += vel_vertical
        
    def _update_ego_lane_idx(self) -> None:
        self.ego_lane_idx = floor((self.ego_pos_y + self.map_height / 2) / self.lane_height)
        return None


    def _update_sensor_max(self, request: RaceCarPredictRequestDto) -> None:
        for name in request.sensors.keys():
            angle = SENSOR_NAME_ANGLE[name]
            
            # If front or back, avoid division by zero
            if abs(angle % pi) < 0.001:
                dist = None
            # Else, calculate distance to wall
            else:
                dist = (self.map_height/2 - sgn(sin(angle))*self.ego_pos_y) / sin(angle)
                dist = abs(dist)
                dist = None if dist > SENSOR_MAX_DISTANCE else dist

            self.sensor_max[name] = dist

    def _update_car_pos_y(self) -> None:
        for i in range(self.lane_count):
            near, far = self.car_pos_y_near[i], self.car_pos_y_far[i]

            if abs(self.ego_pos_y - near) > abs(self.ego_pos_y - far):
                self.car_pos_y_near[i], self.car_pos_y_far[i] = far, near


    def _estimate_car_lane_states(self, request: RaceCarPredictRequestDto) -> None:
        car_pos_x = deepcopy(self.car_pos_x)

        # If there is data from previous time steps, propagate car positions
        if self.request_prev is not None:
            # Dead reckoning estimate horizontal travel distance
            # ego_pos_x_delta = 0.0
            # ego_vel_x = float(self.request_prev.velocity["x"])
            # for action in self.response_prev.actions:
            #     if action == Actions.ACCELERATE:
            #         ego_vel_x += self.accel_left
            #     elif action == Actions.DECELERATE:
            #         ego_vel_x += self.accel_right

            #     ego_pos_x_delta += ego_vel_x


            # Propagate car position information from last request
            for i in range(self.lane_count):
                # If no car position, nothing to do
                car_bounds = car_pos_x[i]
                if car_bounds is None:
                    continue

                # If no velocity estimate, too hard to propagate, so just reset
                if car_bounds.velocity is None:
                    car_pos_x[i] = None
                    continue

                # If guess is too wide, reset it
                # NOTE: This is a heuristic to quickly remove cars that are far away
                if car_bounds.right - car_bounds.left > self.car_guess_max_width + PIXEL_EPSILON:
                    car_pos_x[i] = None
                    continue

                # Not known until a sensor hits it
                car_bounds.known = False


                # NOTE: Because this is a permanent handler, the previous request contains the previous state update time

                # Worst case scenarios of full acceleration in both directions
                # NOTE: Upper and lower bounds can be made if a sensor cuts off the worst case scenario
                #  in one of the ends. But this is complicated and has not been implemented.
                t_prev = self.request_prev.elapsed_ticks - car_bounds.velocity.tick_observed
                
                
                
                
                
                
                
                
                
                # TODO: Fix the speed propagation stuff
                # t_prev = 0
                
                
                
                
                
                
                
                
                
                # NOTE: Adding fudge term of +- 1 because positions are quantized.
                car_vel_prev_forward = car_bounds.velocity.x + 1 + self.car_accel_x*t_prev
                car_vel_prev_backward = car_bounds.velocity.x - 1 - self.car_accel_x*t_prev

                t = request.elapsed_ticks - self.request_prev.elapsed_ticks 
                accel = self.car_accel_x*(t**2)/2
                
                
                
                
                # TODO: Fix the speed propagation stuff
                # accel = self.car_accel_x*((t*0.1)**2)/2
                
                
                
                
                

                car_bounds.right += car_vel_prev_forward*t + accel
                car_bounds.left += car_vel_prev_backward*t - accel

                car_pos_x[i] = car_bounds
        # Else, start with nothing known
        else:
            car_pos_x = [None] * self.lane_count



        # Store areas sensors guarantee are clear
        lane_clears: list[list[tuple[float]]] = [[] for _ in range(self.lane_count)]

        # Process sensor information
        for name, dist in request.sensors.items():
            dist_capped = dist or 1000.0
            angle = SENSOR_NAME_ANGLE[name]

            # Get sensor end point
            sensor_end_x = cos(angle) * dist_capped
            sensor_end_y = sin(angle) * dist_capped + self.ego_pos_y

            # Calculate lane index based on sensor end point
            car_bottom, car_top = min(sensor_end_y, self.ego_pos_y), max(sensor_end_y, self.ego_pos_y)
            lane_pos_bottom = (car_bottom + self.map_height / 2) / self.lane_height
            lane_pos_top = (car_top + self.map_height / 2) / self.lane_height
            # If nummerical issues, sensor is on the same lane
            if abs(lane_pos_bottom - lane_pos_top) < 0.01:
                lane_idx_bottom = floor((self.ego_pos_y + self.map_height / 2) / self.lane_height)
                lane_idx_top = lane_idx_bottom
            # Else, normal processing
            else:
                lane_idx_bottom = max(0, floor((car_bottom + self.map_height / 2) / self.lane_height))
                lane_idx_top = min(self.lane_count - 1, floor((car_top + self.map_height / 2) / self.lane_height))

            if self.ego_pos_y < sensor_end_y:
                lane_idx_clear = list(range(lane_idx_bottom, lane_idx_top))
                lane_idx_start = lane_idx_bottom
                lane_idx_end = lane_idx_top
            else:
                lane_idx_clear = list(range(lane_idx_top, lane_idx_bottom, -1))
                lane_idx_start = lane_idx_top
                lane_idx_end = lane_idx_bottom
                
            # If sensor not fully in first lane, pop it
            # ego_lane_idx = min(self.lane_count, max(0, floor((self.ego_pos_y + self.map_height / 2) / self.lane_height)))
            car_bottom_start = min(self.car_pos_y_near[lane_idx_start], self.car_pos_y_far[lane_idx_start])
            car_top_start = max(self.car_pos_y_near[lane_idx_start], self.car_pos_y_far[lane_idx_start])
            sensor_points_up = self.ego_pos_y < sensor_end_y
            if name not in ["front", "back"] and ((self.ego_pos_y < car_bottom_start and not sensor_points_up) or (self.ego_pos_y > car_top_start and sensor_points_up)):
                lane_idx_clear.remove(lane_idx_start)


            # If sensor hit a car, estimate its position
            dist_hit_car = dist and (dist - (self.sensor_max[name] or inf)) < -PIXEL_EPSILON
            car_bounds = car_pos_x[lane_idx_end]
            if dist_hit_car and not (car_bounds and car_bounds.known):
                car_top = max(self.car_pos_y_near[lane_idx_end], self.car_pos_y_far[lane_idx_end])
                car_bottom = min(self.car_pos_y_near[lane_idx_end], self.car_pos_y_far[lane_idx_end])

                # If hit the car side, guess car position
                # NOTE: Adding an aditional fudge term because of numerical issues
                if abs(sensor_end_y - self.car_pos_y_near[lane_idx_end]) < PIXEL_EPSILON + 1:
                    left = sensor_end_x - self.car_size[0]
                    right = sensor_end_x + self.car_size[0]

                    if car_bounds is None:
                        car_bounds = CarBounds(left, right, known=False)
                    else:
                        car_bounds.left = max(car_bounds.left, left)
                        car_bounds.right = min(car_bounds.right, right)
                # If hit the car front or back, set known car position
                elif car_bottom + PIXEL_EPSILON < sensor_end_y < car_top - PIXEL_EPSILON:
                    # Instantiate new car bounds
                    car_bounds = CarBounds(known=True)

                    # NOTE: Left and right sensors will never hit the front or back.
                    hit_front_of_car = cos(angle) < 0
                    
                    if hit_front_of_car:
                        car_bounds.left = sensor_end_x - self.car_size[0]
                        car_bounds.right = sensor_end_x
                    else:
                        car_bounds.left = sensor_end_x
                        car_bounds.right = sensor_end_x + self.car_size[0]

                    # If car was known, estimate velocity
                    car_bounds_prev = self.car_pos_x[lane_idx_end]
                    if car_bounds_prev and car_bounds_prev.known:
                        distance_prev = self.request_prev.distance if self.response_prev else 0.0
                        ego_vel_x_avg = (request.distance - self.request_prev.distance) / (request.elapsed_ticks - self.request_prev.elapsed_ticks)
                        car_bounds.velocity = CarBounds.Velocity(
                            # NOTE: The estimate is an average velocity, so if long time between samples, things may be very very wrong.
                            x=((car_bounds.right + request.distance) - (car_bounds_prev.right + distance_prev)) / (request.elapsed_ticks - self.request_prev.elapsed_ticks) - ego_vel_x_avg,
                            tick_observed=request.elapsed_ticks
                        )

                # Update car bounds
                car_pos_x[lane_idx_end] = car_bounds
            # Else if angled sensor and inside lane, add sensor as clear area
            elif name not in ["front", "back"] and ((not sensor_points_up and sensor_end_y < self.car_pos_y_near[lane_idx_end] - PIXEL_EPSILON) or (sensor_points_up and sensor_end_y > self.car_pos_y_near[lane_idx_end] + PIXEL_EPSILON)):
                lane_idx_clear.append(lane_idx_end)


            # Sensor did not hit a car in these lanes, update the lane state
            for lane_idx in lane_idx_clear:
                # If sensor is vertical, mitigate numerical issue
                if name in ["left_side", "right_side"]:
                    sensor_left, sensor_right = 0, 0
                elif name == "front":
                    sensor_left, sensor_right = 0, dist_capped
                elif name == "back":
                    sensor_left, sensor_right = -dist_capped, 0
                else:
                    top = max(self.car_pos_y_near[lane_idx], self.car_pos_y_far[lane_idx])
                    bottom = min(self.car_pos_y_near[lane_idx], self.car_pos_y_far[lane_idx])
                    near = sensor_end_y if bottom - PIXEL_EPSILON < sensor_end_y < top + PIXEL_EPSILON else self.car_pos_y_near[lane_idx]
                    far = sensor_end_y if bottom - PIXEL_EPSILON < sensor_end_y < top + PIXEL_EPSILON else self.car_pos_y_far[lane_idx]
                    a = (near - self.ego_pos_y) / tan(angle)
                    b = (far - self.ego_pos_y) / tan(angle)
                    sensor_left, sensor_right = min(a, b), max(a, b)

                left, right = min(sensor_left, sensor_right), max(sensor_left, sensor_right)

                # If processing lane where ego is, ensure sensor does not clear backwards.
                if lane_idx == lane_idx_start:
                    if cos(angle) > 0:
                        left = 0
                    else:
                        right = 0

                # Mark part of lane as clear
                lane_clears[lane_idx].append((left, right))

                # If car bound is too small after update, remove it
                car_pos_x[lane_idx_end] = None if car_bounds and car_bounds.right - car_bounds.left < self.car_size[0] - PIXEL_EPSILON else car_bounds



        
        # Apply clear lane states to car positions
        for lane_idx in range(self.lane_count):
            car_bounds = car_pos_x[lane_idx]

            for left, right in lane_clears[lane_idx]:
                # If no car guess, no information to update
                if car_bounds is None:
                    continue
                # If known from another sensor, no information to add
                if car_bounds.known:
                    continue
                # If not intersecting, no information to update
                if right < car_bounds.left or left > car_bounds.right:
                    continue

                # If slicing the middle of car bounds, figure out which side to keep
                if left > car_bounds.left and right < car_bounds.right:
                    car_width_right = car_bounds.right - right
                    # car_width_left = left - car_bounds.left
                    
                    # NOTE: Biasing it towards guessing cars are in front
                    #  to deal with an edge case where the update is too delayed,
                    #  so the car has had time to move one way or the other.
                    if car_width_right > self.car_size[0] - PIXEL_EPSILON:
                        car_bounds.left = right
                    # elif car_width_left > self.car_size[0] - PIXEL_EPSILON:
                    #     car_bounds.right = left
                    else:
                        car_bounds.right = left
                # Else if slicing the left side, update left bound
                elif left + PIXEL_EPSILON < car_bounds.left < right - PIXEL_EPSILON:
                    car_bounds.left = right
                # Else if slicing the right side, update right bound
                elif left + PIXEL_EPSILON < car_bounds.right < right - PIXEL_EPSILON:
                    car_bounds.right = left
                # Else if within epsilon, ignore
                elif abs(left-car_bounds.right) < PIXEL_EPSILON or abs(right-car_bounds.left) < PIXEL_EPSILON:
                    pass
                # Else, a bug exists
                else:
                    logger.warning(f"Unexpected car bounds update {car_bounds, left, right, lane_idx, name, dist, request.elapsed_ticks}")


                # NOTE: If car bound is too small after update, remove it (should never happen) but it does, so there's a bug.
                # Drive fast in center lane, and then in top lane a car should pass front to back.
                # The known position of the car will be wrong for some reason sometimes.
                # Not fixing, out of time.
                if car_bounds.right - car_bounds.left < self.car_size[0] - PIXEL_EPSILON - 1:
                    logger.warning(f"Unexpected car removal {car_bounds, left, right, lane_idx, request.elapsed_ticks}")
                    car_pos_x[lane_idx] = None


                # Update car bounds
                car_pos_x[lane_idx] = car_bounds



        self.car_pos_x = car_pos_x



    
    
    # TODO: There is a spectrum of lane changes. Fast lane change only sends steering. Use weave in small gaps.
    #  And then distance optimized lane change, which does a small steering input, then accelerates forward while being carried,
    #  and then counter-steers at the end.

    
    # TODO: Potentially widen the maximum car bounds size before removal
    

    
    

    # ===== Main handler =====
    def get_action(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto:
        # Flipping y-axis to match typical math conventions
        request.velocity["y"] = str(-float(request.velocity["y"]))
        
        # Run permanent handlers
        self._update_ticks(request)
        self._update_ego_vel(request)
        self._estimate_ego_pos_y_dead_reckoning()
        self._update_ego_pos_y_wall(request)
        self._update_ego_lane_idx()
        self._update_sensor_max(request)
        self._update_car_pos_y()
        self._estimate_car_lane_states(request)
        
        # Handle request until an action is returned
        response = None
        while response is None and len(self.handler_queue) > 0:
            response = self.handler_queue.popleft()(request)

        response = RaceCarPredictResponseDto(actions=[Actions.NOTHING]) if response is None or len(response.actions) == 0 else response

        # Store old input and output
        self.request_prev = request.model_copy(deep=True)
        self.response_prev = response.model_copy(deep=True)


        # Send action sequence
        return response











    # ===== Visualization =====
    def _get_estimated_ego_center(self, state: GameState) -> tuple[float, float]:
        return (
            state.ego.x + state.ego.sprite.get_width() / 2,
            # Flip y-pos to match pygame's coordinate system
            -self.ego_pos_y + self.map_height / 2 + state.road.lanes[0].y_start
        )
        
    def _get_map_center(self, state: GameState) -> tuple[float, float]:
        return (
            state.ego.x + state.ego.sprite.get_width() / 2,
            self.map_height / 2 + state.road.lanes[0].y_start
        )


    def visualize(self, screen: Surface, state: GameState) -> None:
        # Colors
        color_primary = (255, 0, 255)
        color_secondary = (64, 128, 255)
        color_tertiary = (0, 128, 64)
        
        # Coordinates
        ego_x, ego_y = self._get_estimated_ego_center(state)
        map_x, map_y = self._get_map_center(state)


        # Estimated vertical position
        pygame.draw.circle(screen, color_primary, (ego_x, ego_y), 10)

        # Estimated sensor max
        for name, dist in self.sensor_max.items():
            if dist is not None:
                angle = SENSOR_NAME_ANGLE[name]
                x = dist * cos(angle) + ego_x
                # Flip y-pos to match pygame's coordinate system
                y = -dist * sin(angle) + ego_y

                pygame.draw.circle(screen, color_primary, (x, y), 5)


        # Lane car bounds
        for lane_idx in range(self.lane_count):
            near = self.car_pos_y_near[lane_idx] + map_y
            far = self.car_pos_y_far[lane_idx] + map_y

            pygame.draw.line(
                screen, color_tertiary,
                (map_x - self.car_size[0] / 2, near),
                (map_x + self.car_size[0] / 2, near),
                2
            )
            pygame.draw.line(
                screen, color_tertiary,
                (map_x - self.car_size[0] / 2, far),
                (map_x + self.car_size[0] / 2, far),
                2
            )

        # Visualize car positions
        for lane_idx in range(self.lane_count):
            car_bounds = self.car_pos_x[lane_idx]
            if car_bounds is None:
                continue

            left = car_bounds.left + map_x
            right = car_bounds.right + map_x
            # Flip y-pos to match pygame's coordinate system
            top = -(self.lane_centers[lane_idx] + self.lane_height / 2) + map_y
            bottom = -(self.lane_centers[lane_idx] - self.lane_height / 2) + map_y
            
            # NOTE: It's likely only a visual bug that known cars look too big for one frame.
            pygame.draw.rect(
                screen, color_primary if car_bounds.known else color_secondary,
                (left, top, right - left, bottom - top), 5
            )