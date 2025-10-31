# Drone Home

A full controller for a DJI Mavic 2 Pro drone plus LIDAR sensor which can return the drone safely to the starting point after it is driven manually away from the starting point.
It uses continuous Localisation and Mapping to achieve an understanding of its surroundings.
This data collected is then used to navigate it back automatically.

## Requirements

This uses the existing models for the DJI Mavic 2 Pro drone and the LIDAR sensor on WeBots simulation platform.

We also use existing modules, including WeBots' provided *controller* module to connect to the drone in the simulation, and *numpy* for handling arrays and linear algebra calculations.
