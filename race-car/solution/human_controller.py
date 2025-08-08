import pygame

from solution.controller import Surface, Controller, GameState, RaceCarPredictRequestDto, RaceCarPredictResponseDto


class HumanController(Controller):
    
    def _get_action(self) -> str:
        """
        Reads pygame events and returns an action string based on arrow keys or spacebar.
        Up: ACCELERATE, Down: DECELERATE, Left: STEER_LEFT, Right: STEER_RIGHT, Space: NOTHING
        """
        # Holding down keys
        keys = pygame.key.get_pressed()

        # Priority: accelerate, decelerate, steer left, steer right, nothing
        if keys[pygame.K_RIGHT]:
            return "ACCELERATE"
        if keys[pygame.K_LEFT]:
            return "DECELERATE"
        if keys[pygame.K_UP]:
            return "STEER_LEFT"
        if keys[pygame.K_DOWN]:
            return "STEER_RIGHT"
        if keys[pygame.K_SPACE]:
            return "NOTHING"

        # Just clicking once and it keeps doing it until a new press
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    return "ACCELERATE"
                elif event.key == pygame.K_LEFT:
                    return "DECELERATE"
                elif event.key == pygame.K_UP:
                    return "STEER_LEFT"
                elif event.key == pygame.K_DOWN:
                    return "STEER_RIGHT"
                elif event.key == pygame.K_SPACE:
                    return "NOTHING"
        
        
        # If no relevant key is pressed, repeat last action or do nothing
        #return STATE.latest_action if hasattr(STATE, "latest_action") else "NOTHING"
        return "NOTHING"


    def get_action(self, request: RaceCarPredictRequestDto) -> RaceCarPredictResponseDto:
        return RaceCarPredictResponseDto(actions=[self._get_action()])


    def visualize(self, screen: Surface, state: GameState) -> None:
        return None