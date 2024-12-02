"""
Helmers' Week 5


"""
from math import cos
from math import sin

from controller import Supervisor
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


class Controller:
    def __init__(self):
        # Initialize robot
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Set robot and lidar parameters
        self.max_speed = 10.1
        self.stop = False

        self.X, self.Y = 0, 0
        self.X_offset, self.Y_offset = 2.11, 3.76

        self.lidar_offset = 0.202
        self.lidar_FOV = 4.18879
        self.lidar_max_dist = 5.4
        self.theta = 0

        # Initialize mapping parameters
        self.inf_value = 100
        self.angles = None
        self.pixel_map = np.zeros((300, 300))
        self.robot_width = 36
        self.robot_kernel = np.ones((self.robot_width, self.robot_width))

        # Define waypoints for navigation
        self.waypoint_index = 0
        self.waypoints = [
            [0.53, -0.37], [0.53, -2.5], [0.054, -2.96],
            [-1.34, -2.96], [-1.7, -2.34], [-1.7, -0.54],
            [-1.17, 0.2], [0, 0.2]
        ]
        self.waypoints += [w for w in self.waypoints[-2::-1]] + [[0, 0]]

        # Initialize motors
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(self.max_speed)
        self.rightMotor.setVelocity(self.max_speed)

        # Initialize sensors
        self.lidar = self.robot.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        self.display = self.robot.getDevice("display")

        # Initialize marker for visualization
        self.marker = self.robot.getFromDef("marker").getField("translation")
        self.marker.setSFVec3f(self.waypoints[self.waypoint_index] + [0])

    def update_pos_rot(self):
        # Update robot's position and orientation
        self.X, self.Y, _ = self.gps.getValues()

        # Sanity check
        print(self.gps.getValues())

        compass_values = self.compass.getValues()
        self.theta = np.arctan2(compass_values[0], compass_values[1])

    def read_lidar(self):
        # Fetch lidar data and handle invalid ranges
        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges == np.inf] = self.inf_value

        # Initialize angles if not already set
        if self.angles is None:
            self.angles = np.linspace(self.lidar_FOV / 2, -self.lidar_FOV / 2, len(ranges))[80:-80]

        # Trim obstructed measurements
        ranges = ranges[80:-80]

        # Apply lidar offset to the robot position
        xoffset = self.lidar_offset * np.cos(self.theta)
        yoffset = self.lidar_offset * np.sin(self.theta)

        # Rotation and translation matrix for transforming to world coordinates
        rmatrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta), self.X + xoffset],
            [np.sin(self.theta), np.cos(self.theta), self.Y + yoffset],
            [0, 0, 1]
        ])

        # Convert polar coordinates (ranges and angles) to Cartesian
        xy = np.array([
            ranges * np.cos(self.angles),
            ranges * np.sin(self.angles),
            np.ones(len(ranges))
        ])

        # Apply transformation to world coordinates
        xy_world = rmatrix @ xy

        # Shift coordinates relative to the top-left corner
        xw = xy_world[0, :] + self.X_offset
        yw = xy_world[1, :] + self.Y_offset
        return xw, yw

    def update_display(self, xw, yw):
        # Update pixel map and display lidar data
        xd = np.clip(xw / self.lidar_max_dist * 300, 0, 299).astype(int)
        yd = np.clip((1 - yw / self.lidar_max_dist) * 300, 0, 299).astype(int)

        self.display.setColor(0xFFFFFF)
        self.display.setOpacity(0.01)

        for i in range(len(xd)):
            self.pixel_map[xd[i], yd[i]] = min(1, self.pixel_map[xd[i], yd[i]] + 0.01)
            if self.pixel_map[xd[i], yd[i]] > 0.9:
                self.display.drawPixel(xd[i], yd[i])

        # Draw robot position
        xr = (self.X + self.X_offset) / self.lidar_max_dist * 300
        yr = (1 - (self.Y + self.Y_offset) / self.lidar_max_dist) * 300
        self.display.setColor(0xFF00FF)
        self.display.drawPixel(int(xr), int(yr))

        # Draw waypoint
        xt, yt = self.waypoints[self.waypoint_index]
        xt = (xt + self.X_offset) / self.lidar_max_dist * 300
        yt = (1 - (yt + self.Y_offset) / self.lidar_max_dist) * 300
        self.display.setColor(0xFF0000)
        self.display.drawOval(int(xt), int(yt), 3, 3)

    def move_to_waypoint(self, x, y):
        # Compute distance and angle to waypoint
        dist = np.sqrt((x - self.X)**2 + (y - self.Y)**2)
        target_angle = np.arctan2(y - self.Y, x - self.X) - self.theta
        target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

        # Calculate motor velocities
        rot_scale = target_angle / np.pi
        fwd_scale = 1 - (abs(rot_scale) * 2)**0.2
        velL = self.max_speed * np.clip(-rot_scale + fwd_scale, -1, 1)
        velR = self.max_speed * np.clip(rot_scale + fwd_scale, -1, 1)
        self.leftMotor.setVelocity(velL)
        self.rightMotor.setVelocity(velR)

        return dist < 0.08

    def step(self):
        # Perform a single control step
        self.update_pos_rot()
        xw, yw = self.read_lidar()
        self.update_display(xw, yw)

        if self.move_to_waypoint(*self.waypoints[self.waypoint_index]):
            self.waypoint_index += 1

            if self.waypoint_index < len(self.waypoints):
                self.marker.setSFVec3f(self.waypoints[self.waypoint_index] + [0])
            else:
                self.stop = True
                self.leftMotor.setVelocity(0)
                self.rightMotor.setVelocity(0)

    def run(self):
        # Main control loop
        while self.robot.step(self.timestep) != -1:
            if not self.stop:
                self.step()
            else:
                # Display configuration space
                c_space = convolve2d(self.pixel_map, self.robot_kernel, mode="same") > 1
                plt.imshow(c_space, cmap='gray')
                plt.show()
                break


ctrl = Controller()
ctrl.run()
