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

        self.line_width = 3.75

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

        u = self.curr_control*0.95 + self.last_control*0.05

        self.last_control = u 

        return u

    def absolute_pid_calculate(self):

        '''
        位置式PID控制
        '''
        er1 = self.error_list[-1]
        er2 = self.error_list[-2]
        u = self._kp * er1 + self._kd * (er1 - er2) + self._ki * self.sum_e
        u = 0.95*u + self.last_control*0.05
        self.last_control = u

        return u


    def turn_on(self):

        self.pid_mode = True

    def turn_off(self):

        self.pid_mode = False

        self.sum_e = 0 

        # for indx in range(self.error_list.maxlen):
        #     self.error_list.append(0)
            
        self.step = 0 

    # def update(self, theta):

    #     self.theta = theta
    #     if theta > 0:
    #         error = np.pi - theta 
    #     elif theta < 0 :
    #         error = -np.pi - theta
    #     else:
    #         error = 0 

    #     if self.pid_mode:
    #         self.error_list.append(error)
    #         self.sum_e += error
    #         self.step += 1
    def update(self, theta, offset, pos_y, line_index = 2):
        
        self.theta = theta
        if line_index == 2:
            if pos_y <  ((3.75 / 2) + 1):
                error = -offset
            else:
                error = offset
        elif line_index == 0:
            if 3.75*2.5+1 < pos_y:
                error = offset
            else:
                error = -offset
        else:
            error = 0

        self.error_list.append(error)

        if self.pid_mode:

            self.sum_e += error

            self.step += 1

    def initial(self):

        self.last_control = 0 
        self.theta = -np.pi
        self.sum_e = 0 
        for _ in range(self.error_list.maxlen):
            self.error_list.append(0)


        
        
