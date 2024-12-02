from controller import Supervisor
from collections import deque
from math import sin, cos
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

class Ctrl:
    def __init__(self):
        # Initialize robot
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize variables
        self.max_speed = 10.1
        self.stop = False

        self.X = 0
        self.Y = 0
        self.X_offset = 2.11
        self.Y_offset = 3.76

        self.lidar_offset = 0.202
        self.lidar_FOV = 4.18879
        self.lidar_max_dist = 5.4

        self.theta = 0
        self.inf_value = 100
        self.angles = None
        self.pixel_map = np.zeros((300, 300))

        self.robot_width = 36
        self.robot_kernel = np.ones((self.robot_width, self.robot_width))

        self.waypoint_index = 0
        self.waypoints = [[0.53, -0.37],
                          [0.53, -2.5],
                          [0.054, -2.96],
                          [-1.34, -2.96],
                          [-1.7, -2.34],
                          [-1.7, -0.54],
                          [-1.17, 0.2],
                          [0, 0.2]]
        self.waypoints += [w for w in self.waypoints[-2::-1]] + [[0, 0]]

        # Initialize the motors
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')

        # Velocity mode
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        self.leftMotor.setVelocity(self.max_speed)
        self.rightMotor.setVelocity(self.max_speed)

        # Initialize the sensors
        self.lidar = self.robot.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        self.display = self.robot.getDevice("display")

        # Initialize the marker
        self.marker = self.robot.getFromDef("marker").getField("translation")
        self.marker.setSFVec3f(self.waypoints[self.waypoint_index]+[0])

    def update_pos_rot(self):
        # Read position and orientation via compass and GPS
        self.X, self.Y, _ = self.gps.getValues()
        print(self.gps.getValues())
        tmp = self.compass.getValues()
        self.theta = np.arctan2(tmp[0], tmp[1])

    def read_lidar(self):
        # Read the lidar values, setting invalid points to zero,
        # then convert to world coordinates
        ranges = np.array(self.lidar.getRangeImage())

        if self.angles is None:
            # Angles must also be trimmed to remove obstructed measurements
            self.angles = np.linspace(self.lidar_FOV/2, -self.lidar_FOV/2, len(ranges))
            self.angles = self.angles[80:-80]

        ranges[ranges == np.inf] = self.inf_value
        # Trim obstructed measurements
        ranges = ranges[80:-80]

        # The lidar is offset forward, so all measured points are offset too
        xoffset = self.lidar_offset*np.cos(self.theta)
        yoffset = self.lidar_offset*np.sin(self.theta)

        # Rotation and translation matrix
        rmatrix = np.array([[np.cos(self.theta), -np.sin(self.theta), self.X+xoffset],
                            [np.sin(self.theta), np.cos(self.theta), self.Y+yoffset],
                            [0, 0, 1]])

        # Convert polar to cartesian
        xy = np.array([np.cos(self.angles)*ranges, np.sin(self.angles)*ranges, np.ones(len(ranges))])

        # Transform to world coordinates
        xy_w = rmatrix @ xy

        # Shift to reference top-left corner
        xw = xy_w[0, :]+self.X_offset
        yw = xy_w[1, :]+self.Y_offset
        return xw, yw

    def update_display(self, xw, yw):
        # Offset and normalize robot coordinates
        xr = (self.X+self.X_offset)/self.lidar_max_dist
        yr = (self.Y+self.Y_offset)/self.lidar_max_dist

        # Offset and normalize waypoint coordinates
        xt, yt = self.waypoints[self.waypoint_index]
        xt = (xt+self.X_offset)/self.lidar_max_dist
        yt = (yt+self.Y_offset)/self.lidar_max_dist

        # Convert lidar data to map indexes
        xd = np.clip(xw/self.lidar_max_dist*300, 0, 299).astype(int)
        yd = np.clip((1-yw/self.lidar_max_dist)*300, 0, 299).astype(int)

        # Update map data and draw pixels with a value of at least 0.9
        self.display.setColor(0xFFFFFF)
        self.display.setOpacity(0.01)
        for i in range(len(xd)):
            # self.display.drawPixel(xidx, yidx)
            self.pixel_map[xd[i], yd[i]] = min(1, self.pixel_map[xd[i], yd[i]]+0.01)

            if self.pixel_map[xd[i], yd[i]] > 0.9:
                self.display.drawPixel(xd[i], yd[i])

        # Draw robot position
        self.display.setOpacity(1)
        self.display.setColor(0xFF00FF)
        self.display.drawPixel(xr*300, (1-yr)*300)

        # Draw target position
        self.display.setColor(0xFF0000)
        self.display.drawOval(xt*300, (1-yt)*300, 3, 3)

    def move_to_waypoint(self, x, y):
        # Calculate distance and angle error between robot and target
        dist = ((y-self.Y)**2+(x-self.X)**2)**0.5
        target_angle = np.arctan2(y-self.Y, x-self.X)-self.theta

        # Keep angle in +/-180 degrees range
        if target_angle > np.pi:
            target_angle -= 2*np.pi
        elif target_angle < -np.pi:
            target_angle += 2*np.pi

        # How strongly we need to turn
        rot_scale = target_angle/np.pi
        # Try to move forward unless we need to turn sharply
        fwd_scale = 1-(abs(rot_scale)*2)**0.2

        # Calculate wheel rotation velocities and clip within max speed
        velL = self.max_speed*np.clip(-rot_scale+fwd_scale, -1, 1)
        velR = self.max_speed*np.clip(rot_scale+fwd_scale, -1, 1)

        self.leftMotor.setVelocity(velL)
        self.rightMotor.setVelocity(velR)

        # Return whether we're close enough to the waypoint
        return dist < 0.08

    def step(self):
        # Get current position and rotation
        self.update_pos_rot()

        # Get lidar data in world coordinates
        xw, yw = self.read_lidar()

        # Update display and map with new lidar data
        self.update_display(xw, yw)

        # Try to move to next waypoint
        done = self.move_to_waypoint(*self.waypoints[self.waypoint_index])
        if done:
            # Waypoint reached
            try:
                # Set new target, if available
                self.waypoint_index += 1
                self.marker.setSFVec3f(self.waypoints[self.waypoint_index]+[0])
            except IndexError:
                # End of waypoints, so stop
                self.leftMotor.setVelocity(0)
                self.rightMotor.setVelocity(0)
                self.stop = True


    def run(self):
        while self.robot.step(self.timestep) != -1:
            if not self.stop:
                self.step()
            else:
                # Draw configuration space map
                plt.imshow(convolve2d(self.pixel_map, self.robot_kernel, mode="same")>1)
                plt.show()
                break


C = Ctrl()
C.run()