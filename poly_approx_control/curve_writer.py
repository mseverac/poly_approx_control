import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ur_msgs.msg import PointArray  
from geometry_msgs.msg import Point,Pose,PoseArray,TransformStamped
from std_msgs.msg import Bool,Float32MultiArray
import numpy as np
import time

class CurveWriter(Node):
    def __init__(self):
        super().__init__('curve_writer')
     
    
        
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

        self.create_timer(0.3,self.timer_cb)

        

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

    def on_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info('Writer is ON, ready to log data.')
            self.on = True
        else:
            self.get_logger().info('Writer is OFF, stopping logging.')
            self.on = False

    def tcp_right_callback(self, msg):
        self.get_logger().info(
            f"tcp_right: position=({msg.transform.translation.x}, {msg.transform.translation.y}, {msg.transform.translation.z}), "
            f"orientation=({msg.transform.rotation.x}, {msg.transform.rotation.y}, {msg.transform.rotation.z}, {msg.transform.rotation.w})"
        )
        self.tcp_right = msg.transform

    def tcp_left_callback(self, msg):
        self.get_logger().info(
            f"tcp_left: position=({msg.transform.translation.x}, {msg.transform.translation.y}, {msg.transform.translation.z}), "
            f"orientation=({msg.transform.rotation.x}, {msg.transform.rotation.y}, {msg.transform.rotation.z}, {msg.transform.rotation.w})"
        )
        self.tcp_left = msg.transform

    def points3d_callback(self, msg):

        self.points3d_data = np.array(msg.data).reshape(-1, 3)
        #self.get_logger().info(f"Points3D data shape : {self.points3d_data.shape}")

        self.last_points3d_time = self.get_clock().now()
        #self.get_logger().info(f"last_points3d_time: {self.last_points3d_time}")




    """    def listener_callback(self, msg):
        current_time = self.get_clock().now()
        # Vérifie si une seconde s'est écoulée depuis le dernier message écrit
        if (current_time - self.last_write_time).nanoseconds >= 1e9:
            self.get_logger().info('Message reçu, enregistrement...')
            # On écrit chaque point dans le fichier
            for point in msg.points:
                line = f"{point.x}, {point.y}, {point.z}\n"
                self.log_file.write(line)
            self.log_file.write("---\n")  # Séparation entre les messages
            self.log_file.flush()  # Pour forcer l'écriture immédiate
            self.last_write_time = current_time


    def poses_callback(self, msg : PoseArray):
        current_time = self.get_clock().now()
        # Vérifie si une seconde s'est écoulée depuis le dernier message écrit
        if (current_time - self.last_write_time).nanoseconds >= 1e8:
            self.get_logger().info('Message reçu, enregistrement des poses...')
            # On écrit chaque point dans le fichier
            for i,pose in enumerate(msg.poses):
                point = pose.position
                
                line = f"{point.x}, {point.y}, {point.z}\n"
                self.log_file_poses.write(line)

                if i==0 or i==len(msg.poses)-1:
                    orientation = pose.orientation
                    line = f"{orientation.x}, {orientation.y}, {orientation.z}, {orientation.w}\n"
                    self.log_file_poses.write(line)

            self.log_file_poses.write("---\n")  # Séparation entre les messages
            self.log_file_poses.flush()  # Pour forcer l'écriture immédiate
            self.last_write_time = current_time"""


    def timer_cb(self):
        missing_params = []
        if self.tcp_left is None:
            missing_params.append("tcp left")
        if self.tcp_right is None:
            missing_params.append("tcp right")
        if self.points3d_data is None:
            missing_params.append("points3d_data")
        if not self.on:
            missing_params.append("on")


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
            

        


    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CurveWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
