import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ur_msgs.msg import PointArray  
from geometry_msgs.msg import Point,Pose,PoseArray,TransformStamped
from std_msgs.msg import Bool,Float32MultiArray
import numpy as np
import time

class CurveWriterManual(Node):
    def __init__(self):
        super().__init__('curve_writer_manual')
     
    
        
        self.create_subscription(
            Float32MultiArray,
            'points3d',
            self.points3d_callback,
            10
        )

        self.create_subscription(
            Bool,
            '/writer_on',
            self.on_cb,
            10
        )

        self.create_subscription(
            TransformStamped,
            '/tcp_right',
            self.tcp_right_callback,
            10
        )

        self.create_subscription(
            TransformStamped,
            '/tcp_left',
            self.tcp_left_callback,
            10
        )

        self.on = False
        self.points3d_data = None
        self.last_points3d_time = self.get_clock().now()
        self.tcp_left = None
        self.tcp_right = None



        # Ouvre un fichier pour log
        #self.log_file = open('curve_points_log.txt', 'a')  # 'a' pour append à la fin
        self.log_file_poses = open('curve_poses_log.txt', 'a')  # 'a' pour append à la fin


        self.log_file = open('curve_points_datas.txt', 'a')  # 'a' pour append à la fin


        # Initialise le dernier temps d'écriture
        self.last_write_time = self.get_clock().now()

        self.get_logger().info("node started")

    def on_cb(self, msg: Bool):
        self.get_logger().info("in cb")

        missing_params = []
        if self.tcp_left is None:
            missing_params.append("tcp left")
        if self.tcp_right is None:
            missing_params.append("tcp right")
        if self.points3d_data is None:
            missing_params.append("points3d_data")



        if missing_params:
            self.get_logger().info(f"missing : {', '.join(missing_params)} for writing curve data")
                
            time.sleep(0.2)  # Sleep to avoid busy waiting

        else:
            # Log the TCP positions
            left_pos = self.tcp_left.translation
            right_pos = self.tcp_right.translation
            
            right_orientation = self.tcp_right.rotation
            left_orientation = self.tcp_left.rotation

            self.log_file.write("---\n")  # Séparation entre les messages


            line = f"Right Orientation: {right_orientation.x}, {right_orientation.y}, {right_orientation.z}, {right_orientation.w}\n"
            self.log_file.write(line)
            line = f"Right TCP: {right_pos.x}, {right_pos.y}, {right_pos.z}\n"
            self.log_file.write(line)

            for point in self.points3d_data:
                line = f"{point[0],point[1],point[2]}\n"
                self.log_file.write(line)

            line = f"Left TCP: {left_pos.x}, {left_pos.y}, {left_pos.z}\n"
            self.log_file.write(line)
            line = f"Left Orientation: {left_orientation.x}, {left_orientation.y}, {left_orientation.z}, {left_orientation.w}\n"
            self.log_file.write(line)
            
            self.log_file.flush()
        

    def tcp_right_callback(self, msg):
        
        self.tcp_right = msg.transform

    def tcp_left_callback(self, msg):
        
        self.tcp_left = msg.transform

    def points3d_callback(self, msg):

        self.points3d_data = np.array(msg.data).reshape(-1, 3)
        #self.get_logger().info(f"Points3D data shape : {self.points3d_data.shape}")

        self.last_points3d_time = self.get_clock().now()
        #self.get_logger().info(f"last_points3d_time: {self.last_points3d_time}")


    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CurveWriterManual()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
