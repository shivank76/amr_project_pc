#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan


def clamp_angle(theta):
    """Wrap angle to [-pi, pi]."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(q: Quaternion) -> float:
    # roll/pitch not needed; compute yaw from quaternion
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


def yaw_to_quat(yaw: float) -> Quaternion:
    # roll = pitch = 0
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return Quaternion(x=0.0, y=0.0, z=sy, w=cy)


class Particle:
    __slots__ = ("x", "y", "yaw", "w")
    def __init__(self, x: float, y: float, yaw: float, w: float):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.w = w


class MonteCarloLocalizer(Node):
    def __init__(self):
        super().__init__("task_2")

        # ---- Parameters (override via ROS params if needed)
        self.declare_parameter("num_particles", 200)
        self.declare_parameter("odom_xy_noise", 0.01)     # m std
        self.declare_parameter("odom_yaw_noise", 0.01)    # rad std
        self.declare_parameter("beam_sigma", 0.25)        # m, measurement model
        self.declare_parameter("ray_step", 0.05)          # m, raycast step
        self.declare_parameter("beam_stride", 4)          # use every Nth beam to speed up
        self.declare_parameter("occ_threshold", 50)        # occupancy threshold [0..100]
        self.declare_parameter("use_uniform_init", True)   # else: around /initialpose
        self.declare_parameter("init_std_xy", 0.05)        # m
        self.declare_parameter("init_std_yaw", 0.05)       # rad

        self.N = int(self.get_parameter("num_particles").value)
        self.odom_xy_noise = float(self.get_parameter("odom_xy_noise").value)
        self.odom_yaw_noise = float(self.get_parameter("odom_yaw_noise").value)
        self.beam_sigma = float(self.get_parameter("beam_sigma").value)
        self.ray_step = float(self.get_parameter("ray_step").value)
        self.beam_stride = int(self.get_parameter("beam_stride").value)
        self.occ_thr = int(self.get_parameter("occ_threshold").value)
        self.use_uniform_init = bool(self.get_parameter("use_uniform_init").value)
        self.init_std_xy = float(self.get_parameter("init_std_xy").value)
        self.init_std_yaw = float(self.get_parameter("init_std_yaw").value)

        # ---- State
        self.map = None                     # dict with width,height,resolution,origin,data (np.int8)
        self.last_odom_pose: Pose = None    # Pose from last /odom for delta
        self.scan: LaserScan = None
        self.particles: list[Particle] = []
        self.have_initial_pose = False
        self.init_pose: Pose = None

        # ---- QoS for /map (latched)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---- Subs
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 10)

        # ---- Pubs
        self.pub_particles = self.create_publisher(PoseArray, "/particlecloud", 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/mcl_pose", 10)

        # ---- Timer for visualization
        self.create_timer(0.10, self.publish_particles)   # 10 Hz

    # ==================== Callbacks ====================

    def on_map(self, msg: OccupancyGrid):
        self.map = {
            "w": msg.info.width,
            "h": msg.info.height,
            "res": msg.info.resolution,
            "ox": msg.info.origin.position.x,
            "oy": msg.info.origin.position.y,
            "data": np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        }
        self.get_logger().info("Map received.")

        # If we want uniform init and not yet initialized, initialize here
        if self.use_uniform_init and len(self.particles) == 0:
            self.uniform_initialize()

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        self.init_pose = msg.pose.pose
        self.have_initial_pose = True
        self.gaussian_initialize()
        self.get_logger().info("Particles initialized around /initialpose.")

    def on_scan(self, msg: LaserScan):
        self.scan = msg

    def on_odom(self, msg: Odometry):
        if self.map is None or self.scan is None:
            # Wait until map and scan available
            self.last_odom_pose = msg.pose.pose
            return

        # Ensure particles exist (if no /initialpose yet and uniform init disabled, fall back to uniform)
        if len(self.particles) == 0:
            if self.have_initial_pose:
                self.gaussian_initialize()
            else:
                self.uniform_initialize()

        # Compute odom delta
        if self.last_odom_pose is None:
            self.last_odom_pose = msg.pose.pose
            return

        dx, dy, dyaw = self.pose_delta(self.last_odom_pose, msg.pose.pose)
        self.last_odom_pose = msg.pose.pose

        # If no significant motion, skip heavy update
        if abs(dx) < 1e-4 and abs(dy) < 1e-4 and abs(dyaw) < 1e-4:
            return

        # 1) Prediction (motion update with noise)
        self.motion_update(dx, dy, dyaw)

        # 2) Measurement update (weighting)
        self.measurement_update()

        # 3) Normalize and resample
        self.normalize_weights()
        self.systematic_resample()

        # 4) Publish pose estimate
        self.publish_pose_estimate()

    # ==================== Initialization ====================

    def uniform_initialize(self):
        """Spread particles uniformly over free space cells of the map."""
        if self.map is None:
            return
        free_cells = np.argwhere(self.map["data"] < self.occ_thr)  # rows(y), cols(x)
        if free_cells.size == 0:
            self.get_logger().warn("No free cells in map for uniform init.")
            return
        idx = np.random.choice(free_cells.shape[0], size=self.N, replace=True)
        chosen = free_cells[idx]  # [N, 2] (row, col)

        # Convert to world coords (center of cells)
        res, ox, oy = self.map["res"], self.map["ox"], self.map["oy"]
        xs = ox + (chosen[:, 1] + 0.5) * res
        ys = oy + (chosen[:, 0] + 0.5) * res
        yaws = np.random.uniform(-math.pi, math.pi, size=self.N)

        w0 = 1.0 / self.N
        self.particles = [Particle(float(x), float(y), float(yaw), w0) for x, y, yaw in zip(xs, ys, yaws)]
        self.get_logger().info(f"Uniformly initialized {self.N} particles over free space.")

    def gaussian_initialize(self):
        """Initialize particles around /initialpose with small Gaussian noise."""
        if self.init_pose is None:
            return
        cx = self.init_pose.position.x
        cy = self.init_pose.position.y
        cyaw = quat_to_yaw(self.init_pose.orientation)

        xs = np.random.normal(cx, self.init_std_xy, size=self.N)
        ys = np.random.normal(cy, self.init_std_xy, size=self.N)
        yaws = np.random.normal(cyaw, self.init_std_yaw, size=self.N)
        w0 = 1.0 / self.N
        self.particles = [Particle(float(x), float(y), clamp_angle(float(yaw)), w0) for x, y, yaw in zip(xs, ys, yaws)]

    # ==================== Core PF steps ====================

    def pose_delta(self, p0: Pose, p1: Pose):
        dx = p1.position.x - p0.position.x
        dy = p1.position.y - p0.position.y
        yaw0 = quat_to_yaw(p0.orientation)
        yaw1 = quat_to_yaw(p1.orientation)
        dyaw = clamp_angle(yaw1 - yaw0)
        return dx, dy, dyaw

    def motion_update(self, dx, dy, dyaw):
        # Apply odom delta in world frame with additive Gaussian noise
        for p in self.particles:
            # rotate odom translation into particle's heading frame? Simpler: apply in world
            nx = dx + np.random.normal(0.0, self.odom_xy_noise)
            ny = dy + np.random.normal(0.0, self.odom_xy_noise)
            nyaw = dyaw + np.random.normal(0.0, self.odom_yaw_noise)
            p.x += nx
            p.y += ny
            p.yaw = clamp_angle(p.yaw + nyaw)

    def is_occupied_world(self, x, y) -> bool:
        """Check occupancy at world (x,y)."""
        res, ox, oy = self.map["res"], self.map["ox"], self.map["oy"]
        col = int((x - ox) / res)
        row = int((y - oy) / res)
        if 0 <= row < self.map["h"] and 0 <= col < self.map["w"]:
            return self.map["data"][row, col] >= self.occ_thr
        return True  # outside map -> treat as occupied

    def raycast_range(self, x, y, theta, range_max):
        """Simple grid raycast until hit or max range."""
        s = self.ray_step
        dist = 0.0
        while dist < range_max:
            dist += s
            wx = x + dist * math.cos(theta)
            wy = y + dist * math.sin(theta)
            if self.is_occupied_world(wx, wy):
                return dist
        return range_max

    def measurement_update(self):
        if self.scan is None:
            return
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        rmin = self.scan.range_min
        rmax = self.scan.range_max

        stride = max(1, self.beam_stride)
        sigma = self.beam_sigma
        inv_norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)

        # Precompute beam angles (relative)
        beam_ids = range(0, len(self.scan.ranges), stride)

        for p in self.particles:
            # Likelihood under independent Gaussian errors (in log space to avoid underflow)
            log_w = 0.0
            for i in beam_ids:
                z = self.scan.ranges[i]
                if not (rmin <= z <= rmax) or math.isinf(z) or math.isnan(z):
                    continue
                beam_theta = p.yaw + angle_min + i * angle_inc
                z_hat = self.raycast_range(p.x, p.y, beam_theta, rmax)
                e = z - z_hat
                # log of Gaussian pdf
                log_w += math.log(max(inv_norm * math.exp(-0.5 * (e * e) / (sigma * sigma)), 1e-12))
            p.w = math.exp(log_w) if log_w > -1e12 else 1e-12  # clamp tiny

    def normalize_weights(self):
        s = sum(p.w for p in self.particles)
        if s <= 0.0 or not math.isfinite(s):
            # reset to uniform
            w0 = 1.0 / self.N
            for p in self.particles:
                p.w = w0
            return
        inv = 1.0 / s
        for p in self.particles:
            p.w *= inv

    def systematic_resample(self):
        """Systematic resampling to reduce variance."""
        w = np.array([p.w for p in self.particles], dtype=np.float64)
        cdf = np.cumsum(w)
        cdf[-1] = 1.0
        step = 1.0 / self.N
        r0 = np.random.uniform(0.0, step)
        idx = 0
        new_particles = []
        for m in range(self.N):
            u = r0 + m * step
            while u > cdf[idx]:
                idx += 1
            sel = self.particles[idx]
            # Small jitter to avoid degeneracy
            x = sel.x + np.random.normal(0.0, 0.001)
            y = sel.y + np.random.normal(0.0, 0.001)
            yaw = clamp_angle(sel.yaw + np.random.normal(0.0, 0.001))
            new_particles.append(Particle(x, y, yaw, 1.0 / self.N))
        self.particles = new_particles

    # ==================== Publishing ====================

    def estimate_pose(self):
        """Weighted mean pose (with circular mean for yaw)."""
        if len(self.particles) == 0:
            return None
        ws = np.array([p.w for p in self.particles])
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        yaws = np.array([p.yaw for p in self.particles])
        wsum = np.sum(ws)
        if wsum <= 0:
            return None
        x = float(np.sum(ws * xs) / wsum)
        y = float(np.sum(ws * ys) / wsum)
        # circular mean for yaw
        cy = float(np.sum(ws * np.cos(yaws)) / wsum)
        sy = float(np.sum(ws * np.sin(yaws)) / wsum)
        yaw = math.atan2(sy, cy)
        pose = Pose()
        pose.position = Point(x=x, y=y, z=0.0)
        pose.orientation = yaw_to_quat(yaw)
        return pose

    def publish_pose_estimate(self):
        pose = self.estimate_pose()
        if pose is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose = pose
        # Simple covariance (tune as needed)
        cov_xy = 0.05 * 0.05
        cov_yaw = 0.1 * 0.1
        cov = [0.0] * 36
        cov[0] = cov_xy     # x
        cov[7] = cov_xy     # y
        cov[35] = cov_yaw   # yaw (row 5,col 5)
        msg.pose.covariance = cov
        self.pub_pose.publish(msg)

    def publish_particles(self):
        if len(self.particles) == 0:
            return
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = "map"
        pa.poses = []
        for p in self.particles:
            pose = Pose()
            pose.position = Point(x=p.x, y=p.y, z=0.0)
            pose.orientation = yaw_to_quat(p.yaw)
            pa.poses.append(pose)
        self.pub_particles.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = MonteCarloLocalizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
