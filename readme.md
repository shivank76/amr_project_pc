# Autonomous Mobile Robot – ROS 2

This repository contains ROS 2 Python nodes implementing **planning**, **localization**, and **exploration** for an autonomous mobile robot.

---

## Task 1: Global Path Planning & Control

**Purpose:** Plan a collision-free path to a goal and drive the robot along it.

**Key Points**
- A* global path planning on an inflated occupancy grid
- Path simplification into waypoints
- Waypoint tracking with final orientation alignment

**Topics**
- Subscribed: `/map`, `/goal_pose`, `/scan`, TF
- Published: `/cmd_vel`, `/path`, `/processed_path`

---

## Task 2: Monte Carlo Localization (MCL)

**Purpose:** Estimate robot pose in the map using a particle filter.

**Key Points**
- Particle filter with odometry + laser scan
- Uniform or `/initialpose`-based initialization
- Ray-casting measurement model
- Publishes pose estimate and particle cloud
- Broadcasts `map → odom` TF

**Topics**
- Subscribed: `/map`, `/odom`, `/scan`, `/initialpose`
- Published: `/mcl_pose`, `/particlecloud`, TF

---

## Task 3: Frontier-Based Exploration

**Purpose:** Autonomously explore unknown environments.

**Key Points**
- Detects frontiers (free cells next to unknown)
- Selects frontier goals periodically
- Uses Nav2 `NavigateToPose` action

**Topics**
- Subscribed: `/map`
- Action: `navigate_to_pose`

---

## System Overview

1. **Task 2** localizes the robot.
2. **Task 3** explores and builds the map.
3. **Task 1** plans and executes navigation goals.

This forms a complete **ROS 2 autonomous navigation pipeline**.
