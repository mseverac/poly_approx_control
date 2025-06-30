import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
from std_msgs.msg import Bool


class VisualServoingController(Node):
    def __init__(self):
        super().__init__('visual_servoing_controller')

        # Subscribers
        self.left_vis_vel_sub = self.create_subscription(
            Twist, "/left/vis_vel_cmd_6dof", self.left_vis_vel_cb, 1
        )
        self.right_vis_vel_sub = self.create_subscription(
            Twist, "/right/vis_vel_cmd_6dof", self.right_vis_vel_cb, 1
        )

        # Publishers
        self.left_vis_vel_pub = self.create_publisher(Twist, "/left/vis_vel_cmd_6dof", 1)
        self.right_vis_vel_pub = self.create_publisher(Twist, "/right/vis_vel_cmd_6dof", 1)

        self.writer_pub = self.create_publisher(Bool,"/writer_on",1)

        # Initial state
        self.initial_left_twist = Twist()
        self.initial_right_twist = Twist()

        # Start the movement sequence
        self.move_in_degrees_of_freedom()

    def left_vis_vel_cb(self, msg):
        self.initial_left_twist = msg

    def right_vis_vel_cb(self, msg):
        self.initial_right_twist = msg

    def move_in_degrees_of_freedom(self):

        kl = 0.015
        ka = 0.12

        t = 5  # Time to move in each direction
        ta = 5.0
        # Define the movement steps
        movements = [
                # Mouvements positifs

                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": ka, "angular_y": 0.0, "angular_z": 0.0,"angular ": True},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": ka, "angular_z": 0.0,"angular ": True},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": ka,"angular ": True},
                
                
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": -ka, "angular_y": 0.0, "angular_z": 0.0,"angular ": True},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": -ka, "angular_z": 0.0,"angular ": True},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": -ka,"angular ": True},

                {"linear_x": kl/2, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},
                {"linear_x": 0.0, "linear_y": kl, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": kl, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},


                {"linear_x": -kl/2, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},
                {"linear_x": 0.0, "linear_y": -kl, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},
                {"linear_x": 0.0, "linear_y": 0.0, "linear_z": -kl, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,"angular ": False},



            ]
        
        self.get_logger().info("Starting movement sequence...")
        time.sleep(1)  # Wait for the system to stabilize
        # Publish the writer_on message

        for i in range(10):
            self.writer_pub.publish(Bool(data=True))
        self.get_logger().info("Writer is ON, starting movements...")

        for movement in movements:



            move0 = {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0, "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0}
            # Move left and right
            self.publish_left_movement(movement)
            if movement["angular "]:
                time.sleep(ta)
            else:
                time.sleep(t)
            self.publish_left_movement(move0)
            self.publish_right_movement(movement)
            if movement["angular "]:
                time.sleep(ta)
            else:
                time.sleep(t)            
            self.publish_right_movement(move0)

        self.get_logger().info("Movement sequence completed.")
        # Publish the writer_off message
        for i in range(10):
            self.writer_pub.publish(Bool(data=False))



            

    def publish_left_movement(self, movement):
        # Create Twist message for left
        left_twist = Twist()
        left_twist.linear.x = -movement["linear_x"]
        left_twist.linear.y = movement["linear_y"]
        left_twist.linear.z = movement["linear_z"]
        left_twist.angular.x = movement["angular_x"]
        left_twist.angular.y = movement["angular_y"]
        left_twist.angular.z = movement["angular_z"]
        # Publish the message
        self.left_vis_vel_pub.publish(left_twist)

    def publish_right_movement(self, movement):
        # Create Twist message for right
        right_twist = Twist()
        right_twist.linear.x = movement["linear_x"]
        right_twist.linear.y = movement["linear_y"]
        right_twist.linear.z = movement["linear_z"]
        right_twist.angular.x = movement["angular_x"]
        right_twist.angular.y = movement["angular_y"]
        right_twist.angular.z = movement["angular_z"]
        # Publish the message
        self.right_vis_vel_pub.publish(right_twist)


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()