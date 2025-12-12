# Simplified Drone Control and Autonomous Retrieval

In this project, we made a complete autonomous drone that can be

controlled by the user to fly to any point and can find its own path back to

the starting point. We split the project into 4 rough divisions by integrating user control and stabilisation, localization,

mapping and path planning.

For the drone user controls and stabilisation, we combined sensor data from the IMU and GPS with proportional feedback and motor mixing to create a drone that remains stable whilst still responding quickly to user commands.

To estimate the drone’s position we used

particle filtering method.

For environment, we built a 3D map using lidars

to detect obstacles around it.

With this map, the A\* algorithm calculates a

safe path back home.

Even though each part has slight limitations all these parts (stabilisation, reliable localization,

mapping and autonomous navigation) work almost perfectly together in concentrated areas to make sure the

drone flies smoothly, understands the environment and returns to the start

on its own. However, when traversing larger domains, errors accumulate.

## Requirements

This uses the existing models for the DJI Mavic 2 Pro drone and the LIDAR sensor on WeBots simulation platform.

We also use existing modules, including WeBots' provided *controller* module to connect to the drone in the simulation, and *numpy* for handling arrays and linear algebra calculations.

For a full list of all requirements please refer to the following file: controllers/drone\_controller/requirements.txt



To setup and run this project:



1. Install Webots, make sure Python support is enabled
2. Install required Python version and libraries 
3. Clone this repository 
4. Launch Webots and open world file
5. Verify the controller folder: 

&nbsp;	controllers/
    		drone\_controller/
        		drone\_controller.py
        		Ava\_Drone\_localization.py
        		mapping.py
        		path\_planner.py
        		lidar.py

6\. Ensure Robot controller in scene tree is set to drone\_controller

7\. Run simulation 



Movement controls:



W/S - forward/backward

A/D - left/right

Up/Down arrows - increase/decrease altitude

Left/Right arrows - rotate anti-clockwise/clockwise 



R - Auto-navigate back to origin

M can be used for debugging the map





