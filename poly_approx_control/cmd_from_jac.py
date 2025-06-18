import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray,Header
from geometry_msgs.msg import Point, Twist,TwistStamped
import numpy as np

from ur_msgs.msg import PointArray  
from visualization_msgs.msg import Marker



class Ros2ControllerNode(Node):
    def __init__(self):
        super().__init__('ros2_controller_node')

        self.get_logger().info("Initializing node...")


        #gains pour dlo_nn
        self.k = 2
        self.ka = 0.01
        self.vmax = 0.005

        self.max_time_diff = 0.1  # seconds
        # Subscribers
        self.create_subscription(Float32MultiArray, '/jacobian', self.jacobian_callback, 10)
        self.create_subscription(Float32MultiArray, '/curve_target_6dof', self.curve_target_callback, 10)
        self.create_subscription(Float32MultiArray, "/points3d", self.points3d_callback, 10)

        
        
        # Publishers
        self.cmd_vel_left_pub = self.create_publisher(
            Twist, "/left/vis_vel_cmd_6dof", 1
        )
        self.cmd_vel_right_pub = self.create_publisher(
            Twist, "/right/vis_vel_cmd_6dof", 1
        )

        self.cmd_vel_left_pubs = self.create_publisher(
            TwistStamped, "/left/vis_vel_cmd_6dof_stamped", 1
        )
        self.cmd_vel_right_pubs = self.create_publisher(
            TwistStamped, "/right/vis_vel_cmd_6dof_stamped", 1
        )

        # Data storage
        self.jacobian_data = None
        self.curve_target_data = None
        self.points3d_data = None

        #self.publisher_points_rviz_target = self.create_publisher(Marker, "target_marker", 1)
        self.publisher_points_rviz = self.create_publisher(Marker, "curve_marker", 1)

        self.last_points3d_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.check_timeout)

        self.get_logger().info("Node initialized 2")


    def publish_marker_points_rviz(self, points3d, pub, color=(0.0, 0.0, 1.0), id=0):
        marker = Marker()
        marker.header.frame_id = "cam_bassa_base_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "points"
        marker.id = id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set marker properties
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points3d]

        pub.publish(marker)

        
    def check_timeout(self):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_points3d_time).nanoseconds / 1e9

        if time_diff > self.max_time_diff:
            self.get_logger().error(f"No Points3D data received in the last {time_diff} second. Publishing zero commands.")
            self.get_logger().info(f"now: {current_time}, last: {self.last_points3d_time}, diff: {time_diff}")


            # Publish zero commands
            zero_cmd = Twist()
            self.cmd_vel_left_pub.publish(zero_cmd)
            self.cmd_vel_right_pub.publish(zero_cmd)

    def points3d_callback(self, msg):


        
        self.points3d_data = np.array(msg.data).reshape(-1, 1)
        #self.get_logger().info(f"Points3D data shape : {self.points3d_data.shape}")

        self.last_points3d_time = self.get_clock().now()
        #self.get_logger().info(f"last_points3d_time: {self.last_points3d_time}")


        if self.jacobian_data is None:
            self.get_logger().warn("Jacobian data not received yet.")

        if self.curve_target_data is None:
            self.get_logger().warn("Curve target data not received yet.")
        else:
            # Command computation
            #self.publish_marker_points_rviz(self.curve_target_data.reshape(-1, 3), self.publisher_points_rviz_target)
            self.publish_marker_points_rviz(self.points3d_data.reshape(-1, 3), self.publisher_points_rviz, color=(1.0, 0.0, 0.0))
            s = self.points3d_data.copy()
            sstar = self.curve_target_data.copy()
            ds = sstar - s
            #self.get_logger().info(f"ds shape: {ds.shape}")

            invJ = np.linalg.pinv(self.jacobian_data)
            #self.get_logger().info(f"invJ shape: {invJ.shape}")

            dr = invJ @ ds
            #self.get_logger().info(f"dr shape: {dr.shape}")
            #self.get_logger().info(f"dr: {dr}")

            # Publish the command velocities

            self.pub_cmd_nn(dr)


            """dr = -self.k * dr.astype(float).flatten()


            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "ur_left_gripper"

            cmd_left = Twist()
            cmd_left.linear.x = -dr[0]
            cmd_left.linear.y = dr[8]
            cmd_left.linear.z = dr[4]
            cmd_left.angular.x = -dr[1]*self.ka
            cmd_left.angular.y = dr[9]*self.ka
            cmd_left.angular.z = dr[5]*self.ka
            self.get_logger().info(f"Left command: {cmd_left}")
            self.cmd_vel_left_pub.publish(cmd_left)
            self.cmd_vel_left_pubs.publish(TwistStamped(header=header, twist=cmd_left))

            cmd_right = Twist()
            cmd_right.linear.x = -dr[2]
            cmd_right.linear.y = dr[10]
            cmd_right.linear.z = dr[6]
            cmd_right.angular.x = - dr[3]*self.ka
            cmd_right.angular.y = dr[11]*self.ka
            cmd_right.angular.z = dr[7]*self.ka
            self.get_logger().info(f"Right command: {cmd_right}")
            self.cmd_vel_right_pub.publish(cmd_right)

            header.frame_id = "ur_right_gripper"


            self.cmd_vel_right_pubs.publish(TwistStamped(header=header, twist=cmd_right))
"""
    def pub_cmd_nn(self,dr):
        """Publishes the command velocities to the robot for the nn dr"""

        self.get_logger().info(f"------ Command velocities ---")

        self.get_logger().info(f"norm left linear dr : {np.linalg.norm([dr[0], dr[8], dr[4]])}")
        self.get_logger().info(f"norm right linear dr : {np.linalg.norm([dr[2], dr[10], dr[6]])}")
        self.get_logger().info(f"norm left angular dr : {np.linalg.norm([dr[1], dr[9], dr[5]])}")
        self.get_logger().info(f"norm right angular dr : {np.linalg.norm([dr[3], dr[11], dr[7]])}")

        self.get_logger().info(f"***************")


        dr = self.vmax * np.tanh(-self.k * dr.astype(float).flatten())

        cmd_left = Twist()
        cmd_left.linear.x = -dr[0]
        cmd_left.linear.y = dr[8]
        cmd_left.linear.z = -dr[4]

        cmd_left.angular.x = -dr[1] * self.ka
        cmd_left.angular.y = dr[9] * self.ka
        cmd_left.angular.z = -dr[5] * self.ka

        cmd_right = Twist()
        cmd_right.linear.x = -dr[2]
        cmd_right.linear.y = dr[10]
        cmd_right.linear.z = -dr[6]
        cmd_right.angular.x = -dr[3] * self.ka
        cmd_right.angular.y = dr[11] * self.ka
        cmd_right.angular.z = -dr[7] * self.ka
        #self.get_logger().info(f"Left command: {cmd_left}")
        self.cmd_vel_left_pub.publish(cmd_left)
        #self.cmd_vel_left_pubs.publish(TwistStamped(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="fixed_left_gripper"), twist=cmd_left))
        #self.get_logger().info(f"Right command: {cmd_right}")
        self.cmd_vel_right_pub.publish(cmd_right)
        #self.cmd_vel_right_pubs.publish(TwistStamped(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="fixed_right_gripper"), twist=cmd_right))
        
        self.get_logger().info(f"norm left linear cmd : {np.linalg.norm([cmd_left.linear.x, cmd_left.linear.y, cmd_left.linear.z])}")
        self.get_logger().info(f"norm right linear cmd : {np.linalg.norm([cmd_right.linear.x, cmd_right.linear.y, cmd_right.linear.z])}")
        self.get_logger().info(f"norm left angular cmd : {np.linalg.norm([cmd_left.angular.x, cmd_left.angular.y, cmd_left.angular.z])}")
        self.get_logger().info(f"norm right angular cmd : {np.linalg.norm([cmd_right.angular.x, cmd_right.angular.y, cmd_right.angular.z])}")
        self.get_logger().info(f"***************")

       

    def jacobian_callback(self, msg):
        self.jacobian_data = np.array(msg.data).reshape(-1, 12)
        self.get_logger().info(f"Received Jacobian data: {self.jacobian_data}")

    def curve_target_callback(self, msg):
        self.get_logger().info(f"Received Curve Target message: {msg}")
        self.curve_target_data = np.array(msg.data).reshape(-1, 1)

        self.get_logger().info(f"Processed Curve Target data (shape {self.curve_target_data.shape}): {self.curve_target_data}")
   

def main(args=None):
    rclpy.init(args=args)
    node = Ros2ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()