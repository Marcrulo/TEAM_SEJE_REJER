# Race car
\`\`\`
cd race-car
uv run example.py
\`\`\`

The main duct taped together mess is in solution/queue_controller.py. It is a queue of handlers that decide how the car should drive and what items are in the queue. There is also a bunch of code that infers the state (position and velocity) of the other cars by reasoning about upper and lower bounds of a car's position that are consistent with the observed data - there's also some propagation of state forwards in time. 

Attached is the exact code used for the evaluation, so the results should be reproducible. The code became extremely messy towards the end, because we were simply not done (as evidenced by the many TODOs and the lack of the implementation of the TODO describing the subroutine that would have prevented us from crashing in the evaluation run).