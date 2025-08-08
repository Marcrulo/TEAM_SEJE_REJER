import time
import uvicorn
import datetime
from loguru import logger
from fastapi import Body, FastAPI

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from solution.queue_controller import QueueController

HOST = "0.0.0.0"
PORT = 38080


app = FastAPI()
start_time = time.time()

controller = QueueController()

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    global controller
    if request.elapsed_ticks == 1:
        logger.info("New run started. Resetting controller.")
        controller = QueueController()

    return controller.get_action(request)

@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"




if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
