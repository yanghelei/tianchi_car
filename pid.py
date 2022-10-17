from collections import deque
import numpy as np

class PIDController(object):
    
    def __init__(self, params):

        self._ki = params.ki
        self._kp = params.kp 
        self._kd = params.kd 

        self.error_list = deque(maxlen=3)

        self.last_control = None # 上一次控制量
        self.curr_control = None # 当前控制量
        self.theta = None # 车辆朝向角
        self.step = 0

        self.sum_e = 0

        self.pid_mode = False # 是否开启PID模式

    def incre_pid_calculate(self):

        '''
        增量式PID控制
        '''

        er1 = self.error_list[-1] # 最近一次的误差
        er2 = self.error_list[-2]
        er3 = self.error_list[-3]

        control_delta = (self._kp * (er1 - er2) + self._ki * er1 + \
                    self._kd * (er1 - 2 * er2 + er3))

        if self.last_control is None:
            self.last_control = 0

        self.curr_control = self.last_control + control_delta

        u = self.curr_control*0.9 + self.last_control

        self.last_control = u 

        return u

    def absolute_pid_calculate(self):

        '''
        位置式PID控制
        '''
        er1 = self.error_list[-1]
        er2 = self.error_list[-2]
        u = self._kp * er1 + self._kd * (er1 - er2) + self._ki * self.sum_e

        self.last_control = u

        return u


    def turn_on(self):

        self.pid_mode = True

    def turn_off(self):

        self.pid_mode = False

        self.sum_e = 0 

        for indx in range(len(self.error_list)):
            self.error_list[indx] = 0
            
        self.step = 0 

    def update(self, theta):

        self.theta = theta
        if theta > 0:
            error = np.pi - theta 
        elif theta < 0 :
            error = -np.pi - theta
        else:
            error = 0 

        if self.pid_mode:
            self.error_list.append(error)
            self.sum_e += error
            self.step += 1

    def initial(self):

        self.last_control = 0 
        self.theta = -np.pi
        self.sum_e = 0 
        for indx in range(len(self.error_list)):
            self.error_list[indx] = 0


        
        
