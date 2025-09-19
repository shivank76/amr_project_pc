import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import math
import random

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Subscribe to SLAM map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        # Nav2 action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.map = None
        self.get_logger().info("Frontier Explorer Node started.")

        # Run exploration loop every 10 seconds
        self.timer = self.create_timer(10.0, self.explore_callback)

    def map_callback(self, msg):
        self.map = msg

    def explore_callback(self):
        if self.map is None:
            self.get_logger().warn("No map yet.")
            return

        # Find frontiers
        frontiers = self.detect_frontiers()
        if not frontiers:
            self.get_logger().info("Exploration complete. No frontiers found.")
            return

        # Select a random frontier point
        goal = random.choice(frontiers)
        wx, wy = self.map_to_world(goal[0], goal[1])

        # Send goal to Nav2
        self.send_goal(wx, wy)

    def detect_frontiers(self):
        """Detect frontier cells (unknown adjacent to free)."""
        grid = np.array(self.map.data).reshape((self.map.info.height, self.map.info.width))
        frontiers = []
        for y in range(1, self.map.info.height - 1):
            for x in range(1, self.map.info.width - 1):
                if grid[y, x] == 0:  # free cell
                    neighbors = [grid[y+dy, x+dx] for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
                    if -1 in neighbors:  # unknown nearby
                        frontiers.append((x, y))
        return frontiers

    def map_to_world(self, mx, my):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        wx = mx * res + origin.x
        wy = my * res + origin.y
        return wx, wy

    def send_goal(self, x, y):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0  # face forward

        self.get_logger().info(f"Sending goal: ({x:.2f}, {y:.2f})")
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)

        def goal_response_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Goal rejected!")
                return
            self.get_logger().info("Goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.result_callback)

        send_goal_future.add_done_callback(goal_response_cb)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Goal reached.")


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import PoseStamped, Point
# from nav2_msgs.action import NavigateToPose
# import numpy as np
# import math
# from scipy.ndimage import label

# class FrontierExplorer(Node):
#     def __init__(self):
#         super().__init__('exploration_node')

#         # Subscribers
#         self.map_sub = self.create_subscription(
#             OccupancyGrid, '/map', self.map_callback, 10
#         )

#         # Nav2 action client
#         self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

#         # Internal state
#         self.map = None
#         self.robot_pose = None
#         self.frontiers = []
#         self.map_inflation = 2  # cells to inflate obstacles
#         self.exploring = False

#         self.timer = self.create_timer(1.0, self.timer_callback)

#     # ---------------- Map callback ----------------
#     def map_callback(self, msg):
#         width = msg.info.width
#         height = msg.info.height
#         data = np.array(msg.data, dtype=np.int8).reshape((height, width))
#         self.map = {
#             'data': data,
#             'width': width,
#             'height': height,
#             'res': msg.info.resolution,
#             'origin_x': msg.info.origin.position.x,
#             'origin_y': msg.info.origin.position.y
#         }

#     # ---------------- Timer callback ----------------
#     def timer_callback(self):
#         if self.map is None or self.exploring:
#             return

#         self.frontiers = self.detect_frontiers()
#         if not self.frontiers:
#             self.get_logger().info("Exploration complete! No frontiers left.")
#             return

#         best_frontier = self.select_best_frontier(self.frontiers)
#         self.send_goal(best_frontier)

#     # ---------------- Frontier detection ----------------
#     def detect_frontiers(self):
#         map_data = self.map['data']
#         height, width = map_data.shape
#         frontiers = np.zeros_like(map_data, dtype=bool)

#         # Detect free cells adjacent to unknown cells
#         for y in range(1, height - 1):
#             for x in range(1, width - 1):
#                 if map_data[y, x] == 0:  # free
#                     neighbors = map_data[y-1:y+2, x-1:x+2]
#                     if np.any(neighbors == -1):
#                         frontiers[y, x] = True

#         # Label frontier clusters
#         labeled, num_features = label(frontiers)
#         clusters = []
#         for i in range(1, num_features + 1):
#             ys, xs = np.where(labeled == i)
#             if len(xs) > 0:
#                 cx = np.mean(xs)
#                 cy = np.mean(ys)
#                 clusters.append((cx, cy))

#         return clusters

#     # ---------------- Frontier scoring ----------------
#     def select_best_frontier(self, clusters):
#         best_score = -math.inf
#         best = None
#         robot_pos = self.get_robot_map_position()
#         if robot_pos is None:
#             robot_pos = (0,0)

#         for cx, cy in clusters:
#             dist = math.hypot(cx - robot_pos[0], cy - robot_pos[1])
#             score = 1.0 / (dist + 1e-3)  # closer frontiers are preferred
#             if score > best_score:
#                 best_score = score
#                 best = (cx, cy)
#         return best

#     # ---------------- Map to world ----------------
#     def map_to_world(self, mx, my):
#         x = mx * self.map['res'] + self.map['origin_x'] + self.map['res']/2
#         y = my * self.map['res'] + self.map['origin_y'] + self.map['res']/2
#         return x, y

#     # ---------------- Robot position ----------------
#     def get_robot_map_position(self):
#         # For simplicity, use odometry at origin (or could subscribe to /amcl_pose)
#         # TODO: Replace with actual robot pose
#         return (0, 0)

#     # ---------------- Send goal ----------------
#     def send_goal(self, frontier):
#         if frontier is None:
#             return
#         self.exploring = True
#         mx, my = frontier
#         wx, wy = self.map_to_world(mx, my)

#         goal_pose = PoseStamped()
#         goal_pose.header.frame_id = "map"
#         goal_pose.header.stamp = self.get_clock().now().to_msg()
#         goal_pose.pose.position.x = wx
#         goal_pose.pose.position.y = wy
#         goal_pose.pose.orientation.w = 1.0

#         self.get_logger().info(f"Sending goal to frontier at ({wx:.2f}, {wy:.2f})")

#         goal_msg = NavigateToPose.Goal()
#         goal_msg.pose = goal_pose

#         self.nav_to_pose_client.wait_for_server()
#         self._send_goal_future = self.nav_to_pose_client.send_goal_async(
#             goal_msg,
#             feedback_callback=self.feedback_callback
#         )
#         self._send_goal_future.add_done_callback(self.goal_response_callback)

#     def goal_response_callback(self, future):
#         goal_handle = future.result()
#         if not goal_handle.accepted:
#             self.get_logger().warn("Goal rejected by Nav2!")
#             self.exploring = False
#             return
#         self._get_result_future = goal_handle.get_result_async()
#         self._get_result_future.add_done_callback(self.get_result_callback)

#     def feedback_callback(self, feedback_msg):
#         pass

#     def get_result_callback(self, future):
#         result = future.result().result
#         self.get_logger().info("Reached frontier.")
#         self.exploring = False

# # ---------------- Main ----------------
# def main(args=None):
#     rclpy.init(args=args)
#     node = FrontierExplorer()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()
