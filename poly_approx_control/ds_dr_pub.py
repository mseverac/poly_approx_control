import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TwistStamped



def trans_to_vecs(trans: TransformStamped):
  
    pos = trans.transform.translation
    ori = trans.transform.rotation

    r_curr = R.from_quat([ori.x, ori.y, ori.z, ori.w])

    return r_curr.as_rotvec(),np.array([pos.x, pos.y, pos.z])

def compute_r_from_transes(tr :TransformStamped,tl :TransformStamped):
    """Compute the r vector from the transforms of the left and right end effectors"""
    r_left, pos_left = trans_to_vecs(tl)
    r_right, pos_right = trans_to_vecs(tr)

    r = np.array([pos_right[0],pos_right[1],pos_right[2],
                  r_right[0],r_right[1],r_right[2],
                  pos_left[0],pos_left[1],pos_left[2],
                  r_left[0],r_left[1],r_left[2]], dtype=np.float64)
    
    return r


def decompose_r(r):
    """Decompose the r vector into positions and rotation matrices for left and right end effectors"""
    
    # Extraire les positions
    r_pos = np.array(r[0:3], dtype=np.float64)
    vr = np.array(r[3:6], dtype=np.float64)
    l_pos = np.array(r[6:9], dtype=np.float64)
    vl = np.array(r[9:12], dtype=np.float64)
    
    return r_pos, l_pos, vr, vl

class DsDrPub(Node):
    def __init__(self):
        super().__init__('ds_dr_pub')
        self.subscription_points3d = self.create_subscription(
            Float32MultiArray,
            '/points3d',
            self.points3d_callback,
            10)
        self.subscription_tcp_left = self.create_subscription(
            TransformStamped,
            '/tcp_left',
            self.tcp_left_callback,
            10)
        self.subscription_tcp_right = self.create_subscription(
            TransformStamped,
            '/tcp_right',
            self.tcp_right_callback,
            10)
        

        self.dr_l_pub = self.create_publisher(TwistStamped, 'dr_l', 10)
        self.dr_r_pub = self.create_publisher(TwistStamped, 'dr_r', 10)
        
        self.last_s_pub = self.create_publisher(Float32MultiArray, '/last_s', 10)
        self.s_pub = self.create_publisher(Float32MultiArray, '/s', 10)

        self.dr_pub = self.create_publisher(Float32MultiArray, '/dr', 10)
        self.delta_pub = self.create_publisher(Float32MultiArray, '/s_ls_dr', 10)


        self.last_s_pub_viz = self.create_publisher(Marker, 'last_s_marker', 10)
        self.s_pub_viz = self.create_publisher(Marker, 's_marker', 10)

        
        self.create_timer(0.2,self.timer_cb)


        
        self.s = None
        self.tcp_l = None
        self.tcp_r = None

        self.last_s = None
        self.last_tcp_l = None
        self.last_tcp_r = None

        

        
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


    def points3d_callback(self, msg):
        #self.get_logger().info(f'Received /points3d: {msg.data}')
        self.s = np.array(msg.data)

    def tcp_left_callback(self, msg):
        self.tcp_l = msg

    def tcp_right_callback(self, msg):
        self.tcp_r = msg


    def timer_cb(self):
        missing_params = []
        if self.tcp_l is None:
            missing_params.append("tcp left")
        if self.tcp_r is None:
            missing_params.append("tcp right")
        if self.s is None:
            missing_params.append("points3d_data")
        

        if missing_params:
            self.get_logger().info(f"missing : {', '.join(missing_params)} for writing curve data")

            return
                
        if self.last_s is None : 
            self.last_s = self.s
            self.last_r = compute_r_from_transes(self.tcp_r,self.tcp_l)

            return
        
        r = compute_r_from_transes(self.tcp_r,self.tcp_l)
        dr = r - self.last_r
        
        ds = self.s - self.last_s

        norms_ds = np.array([np.linalg.norm(dp) for dp in ds.reshape(-1,3)])

        max_norm = np.max(norms_ds)



        def viz_ds_dr(s,last_s,dr):

            self.publish_marker_points_rviz(last_s.reshape(-1,3),self.last_s_pub_viz,color=(0.3,0.3,0.3))
            self.publish_marker_points_rviz(s.reshape(-1,3),self.s_pub_viz,color=(0.7,0.7,0.7))

            
            r_pos, l_pos, vr, vl = decompose_r(-dr)

            # Publish dr for right gripper
            dr_r_msg = TwistStamped()
            dr_r_msg.header.stamp = self.get_clock().now().to_msg()
            dr_r_msg.header.frame_id = "fixed_right_gripper_bf"
            dr_r_msg.twist.linear.x = r_pos[0]
            dr_r_msg.twist.linear.y = r_pos[1]
            dr_r_msg.twist.linear.z = r_pos[2]
            dr_r_msg.twist.angular.x = vr[0]
            dr_r_msg.twist.angular.y = vr[1]
            dr_r_msg.twist.angular.z = vr[2]
            self.dr_r_pub.publish(dr_r_msg)

            # Publish dr for left gripper
            dr_l_msg = TwistStamped()
            dr_l_msg.header.stamp = self.get_clock().now().to_msg()
            dr_l_msg.header.frame_id = "fixed_left_gripper_bf"
            dr_l_msg.twist.linear.x = l_pos[0]
            dr_l_msg.twist.linear.y = l_pos[1]
            dr_l_msg.twist.linear.z = l_pos[2]
            dr_l_msg.twist.angular.x = vl[0]
            dr_l_msg.twist.angular.y = vl[1]
            dr_l_msg.twist.angular.z = vl[2]
            self.dr_l_pub.publish(dr_l_msg)

        if max_norm > 0.01 :

            
            self.get_logger().info(f"ds: {ds.reshape(-1,3)}")
            self.get_logger().info(f"norms_ds: {norms_ds}")
            self.get_logger().info(f"max_norm: {max_norm}")
            self.get_logger().info(f"dr : {dr.reshape(-1,1)}")
            viz_ds_dr(self.s,self.last_s,dr)

            """self.get_logger().info(f"s shape: {self.s.shape}")
            self.get_logger().info(f"last_s shape: {self.last_s.shape}")
            self.get_logger().info(f"dr shape: {dr.shape}")
            self.get_logger().info(f"s ls dr shape: {s_ls_dr.shape}")"""

            s_ls_dr = np.concatenate((self.s,self.last_s,dr))
            delta_msg = Float32MultiArray()
            delta_msg.data = s_ls_dr.tolist()
            self.delta_pub.publish(delta_msg)


            s_msg = Float32MultiArray()
            s_msg.data = self.s.tolist()
            self.s_pub.publish(s_msg)

            ls_msg = Float32MultiArray()
            ls_msg.data = self.last_s.tolist()
            self.last_s_pub.publish(ls_msg)

            dr_msg = Float32MultiArray()
            dr_msg.data = dr.tolist()
            self.dr_pub.publish(dr_msg)

            self.last_r = r 
            self.last_s = self.s 




        










def main(args=None):
    rclpy.init(args=args)
    node = DsDrPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()