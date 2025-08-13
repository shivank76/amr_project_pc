import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import math
import heapq
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

class AMRPlannerNode(Node):
    def __init__(self):
        super().__init__('task_1')

        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_path_pub = self.create_publisher(Path, '/path', 10)
        self.processed_path_pub = self.create_publisher(Path, 'processed_path', 10)

        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.map = None
        self.map_array = None
        self.inflation_radius = int(0.4417 / 0.05)

        self.path_waypoints = []
        self.current_waypoint_index = 0
        self.path_received = False
        self.obstacle_points_base_link = []

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.k_a = 3.5
        self.k_r = 0.3
        self.threshold_dis = 1.0
        self.waypoint_tolerance = 0.3
        self.final_goal_tolerance = 0.1
        self.orientation_tolerance = 0.1
        self.lookahead_distance = 1.0
        self.max_waypoint_jump = 3

        self.goal_yaw = None

        self.get_logger().info("AMR Enhanced Planner Node started.")

    def map_callback(self, msg):
        self.map = msg
        self.map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.inflate_map()
        self.get_logger().info("Map received and inflated.")

    def inflate_map(self):
        inflated = self.map_array.copy()
        height, width = self.map_array.shape
        inflation_radius = int(self.inflation_radius * 1.2)  # 20% more space
        for y in range(height):
            for x in range(width):
                if self.map_array[y, x] >= 50:
                    for dy in range(-inflation_radius, inflation_radius + 1):
                        for dx in range(-inflation_radius, inflation_radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                inflated[ny, nx] = 100
        self.map_array = inflated

    def goal_callback(self, msg):
        if self.map is None:
            self.get_logger().warn("Map not received yet.")
            return

        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            start_x = trans.transform.translation.x
            start_y = trans.transform.translation.y
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        quat = msg.pose.orientation
        _, _, self.goal_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        path = self.plan_path(start_x, start_y, goal_x, goal_y)
        if path:
            self.publish_path(path)

    def world_to_map(self, x, y):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        return int((x - origin.x) / res), int((y - origin.y) / res)

    def map_to_world(self, mx, my):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        return mx * res + origin.x, my * res + origin.y

    def is_in_map(self, mx, my):
        return 0 <= mx < self.map.info.width and 0 <= my < self.map.info.height

    def is_occupied(self, mx, my):
        return self.map_array[my, mx] >= 50

    def get_neighbors(self, node):
        x, y = node
        dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]
        return [(x+dx, y+dy) for dx, dy in dirs if self.is_in_map(x+dx, y+dy) and not self.is_occupied(x+dx, y+dy)]

    def heuristic(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def distance(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        start = self.world_to_map(start_x, start_y)
        goal = self.world_to_map(goal_x, goal_y)

        if not self.is_in_map(*start) or not self.is_in_map(*goal):
            self.get_logger().warn("Start or goal outside map bounds.")
            return None
        if self.is_occupied(*start) or self.is_occupied(*goal):
            self.get_logger().warn("Start or goal in occupied cell.")
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    if not any(neighbor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def publish_path(self, path_cells):
        raw_path = Path()
        raw_path.header.frame_id = "map"
        raw_path.header.stamp = self.get_clock().now().to_msg()

        poses = []
        for mx, my in path_cells:
            wx, wy = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        raw_path.poses = poses
        self.global_path_pub.publish(raw_path)
        self.get_logger().info(f"Published raw path with {len(poses)} poses to /path")

        self.postprocess_path(raw_path)

    def postprocess_path(self, msg):
        original_path = msg.poses
        processed_path = []

        if not original_path:
            self.get_logger().warn("Received empty path.")
            return

        processed_path.append(original_path[0])
        for i in range(1, len(original_path) - 1):
            prev = original_path[i - 1].pose.position
            curr = original_path[i].pose.position
            nxt = original_path[i + 1].pose.position

            angle1 = math.atan2(curr.y - prev.y, curr.x - prev.x)
            angle2 = math.atan2(nxt.y - curr.y, nxt.x - curr.x)
            if abs(angle2 - angle1) > 0.4:
                processed_path.append(original_path[i])
        processed_path.append(original_path[-1])

        new_path = Path()
        new_path.header = msg.header
        new_path.poses = processed_path
        self.path_waypoints = [np.array([p.pose.position.x, p.pose.position.y]) for p in processed_path]
        self.current_waypoint_index = 0
        self.path_received = True

        self.processed_path_pub.publish(new_path)
        self.get_logger().info(f"Published processed path with {len(processed_path)} waypoints.")

    def laser_callback(self, msg):
        angle = msg.angle_min
        self.obstacle_points_base_link = []
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                self.obstacle_points_base_link.append(np.array([x, y]))
            angle += msg.angle_increment

    def timer_callback(self):
        if not self.path_received:
            self.stop_robot()
            return

        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y
            quat = trans.transform.rotation
            _, _, self.robot_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            robot_pos = np.array([self.robot_x, self.robot_y])
            if not self.path_waypoints:
                self.stop_robot()
                return

            final_goal = self.path_waypoints[-1]
            dist_to_goal = np.linalg.norm(robot_pos - final_goal)

            if dist_to_goal < self.final_goal_tolerance:
                yaw_error = math.atan2(math.sin(self.goal_yaw - self.robot_yaw), math.cos(self.goal_yaw - self.robot_yaw))
                if abs(yaw_error) < self.orientation_tolerance:
                    self.get_logger().info("Goal reached with orientation.")
                    self.stop_robot()
                    return
                else:
                    twist = Twist()
                    twist.angular.z = max(-1.0, min(1.0, 2.5 * yaw_error))
                    self.cmd_pub.publish(twist)
                    return

            target_wp = self.path_waypoints[self.current_waypoint_index]
            if np.linalg.norm(robot_pos - target_wp) < self.waypoint_tolerance:
                self.current_waypoint_index = min(self.current_waypoint_index + 1, len(self.path_waypoints) - 1)
                target_wp = self.path_waypoints[self.current_waypoint_index]

            dir_to_target = target_wp - robot_pos
            dist = np.linalg.norm(dir_to_target)
            v_attr = self.k_a * (dir_to_target / dist) if dist > 0.01 else np.array([0.0, 0.0])

            velocity = v_attr
            linear = np.linalg.norm(velocity)
            desired_yaw = math.atan2(velocity[1], velocity[0])
            yaw_error = math.atan2(math.sin(desired_yaw - self.robot_yaw), math.cos(desired_yaw - self.robot_yaw))

            twist = Twist()
            if abs(yaw_error) > 0.3:
                twist.angular.z = max(-2.0, min(2.0, 3.0 * yaw_error))
                twist.linear.x = 0.0
            else:
                twist.linear.x = min(linear, 0.7)
                twist.angular.z = max(-1.5, min(1.5, 2.0 * yaw_error))

            self.cmd_pub.publish(twist)

        except TransformException as e:
            self.get_logger().warn(f'TF error: {str(e)}')
            self.stop_robot()
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
            self.stop_robot()

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AMRPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


