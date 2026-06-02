
class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.previous_error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error
        return output
