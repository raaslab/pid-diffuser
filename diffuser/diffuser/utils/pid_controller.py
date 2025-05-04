""" 
Simple PID controller 
"""

import numpy as np
class PIDPositionVelocityController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = np.zeros(2)

    def compute_action(self, desired_state, current_state, dt):
        desired_pos = desired_state[:2]
        desired_vel = desired_state[2:]
        current_pos = current_state[:2]
        current_vel = current_state[2:]
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel

        self.integral_error += pos_error * dt

        action = (self.Kp * pos_error +
                  self.Ki * self.integral_error +
                  self.Kd * vel_error)
        return action