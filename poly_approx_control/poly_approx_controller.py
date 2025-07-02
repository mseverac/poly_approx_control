from geometry_msgs.msg import Point,Pose,PoseArray,Twist,Vector3,TransformStamped,TwistStamped
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.wait_for_message import wait_for_message
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension,Float32MultiArray,Header
import time

def trans_to_matrix(trans: TransformStamped):
  
    pos = trans.transform.translation
    ori = trans.transform.rotation

    r_curr = R.from_quat([ori.x, ori.y, ori.z, ori.w])

    return r_curr.as_matrix(),np.array([pos.x, pos.y, pos.z])

def trans_to_vecs(trans: TransformStamped):
  
    pos = trans.transform.translation
    ori = trans.transform.rotation

    r_curr = R.from_quat([ori.x, ori.y, ori.z, ori.w])

    return r_curr.as_rotvec(),np.array([pos.x, pos.y, pos.z])

def compute_r(tr :TransformStamped,tl :TransformStamped):
    """Compute the r vector from the transforms of the left and right end effectors"""
    r_left, pos_left = trans_to_vecs(tl)
    r_right, pos_right = trans_to_vecs(tr)

    r = np.array([pos_right[0],pos_right[1],pos_right[2],
                  r_right[0],r_right[1],r_right[2],
                  pos_left[0],pos_left[1],pos_left[2],
                  r_left[0],r_left[1],r_left[2]], dtype=np.float64)
    
    return r


class Broyden_controller_s_Ainit(Node):
    def __init__(self):
        super().__init__('Broyden_controller_s_Ainit')


        self.curve_target_data = None

        self.ka_far = 0.1
        self.ka_near = 1
        self.k = 10
        self.vmax = 0.02

        self.s_star = None
        self.A = None


        self.cmd_vel_left_pub = self.create_publisher(
            Twist, "/left/vis_vel_cmd_6dof", 1
        )
        self.cmd_vel_right_pub = self.create_publisher(
            Twist, "/right/vis_vel_cmd_6dof", 1
        )

        self.cmd_vel_left_pubs = self.create_publisher(
            TwistStamped, "/left/cart_vel_cmd_6dof_stamped", 1
        )
        self.cmd_vel_right_pubs = self.create_publisher(
            TwistStamped, "/right/cart_vel_cmd_6dof_stamped", 1
        )

        self.A_pub = self.create_publisher(Float64MultiArray, '/A_Broyden', 10)

    
        self.create_subscription(Float32MultiArray,"/curve_target_6dof",self.curve_target_callback,1)
        self.create_subscription(Float32MultiArray,"/points3d",self.cmd_cb,1)

        
        self.create_subscription(Float64MultiArray,"/A_init",self.A_cb,1)


    def A_cb(self,msg):
        self.A = np.array(msg.data).reshape(153,12)
        A = self.A

        self.get_logger().info(f"First row of A: {A[0]}")
        self.get_logger().info(f"Second row of A: {A[1]}")
        self.get_logger().info(f"Third row of A: {A[2]}")
        self.get_logger().info(f"-3 row of A: {A[-3]}")
        self.get_logger().info(f"-2 row of A: {A[-2]}")
        self.get_logger().info(f"-1 row of A: {A[-1]}")

    def pub_cmd(self,dr):
        """Publishes the command velocities to the robot for the nn dr"""

        self.get_logger().info(f"------ Command velocities ---")

        self.get_logger().info(f"dr: {dr}")

        # dr = [linear_r(0:3), angular_r(3:6), linear_l(6:9), angular_l(9:12)]
        self.get_logger().info(f"norm right linear dr : {np.linalg.norm(dr[0:3])}")
        self.get_logger().info(f"norm right angular dr : {np.linalg.norm(dr[3:6])}")
        self.get_logger().info(f"norm left linear dr : {np.linalg.norm(dr[6:9])}")
        self.get_logger().info(f"norm left angular dr : {np.linalg.norm(dr[9:12])}")

        self.get_logger().info(f"***************")


        self.get_logger().info(f"k : {self.k}")

        if np.linalg.norm(dr[0:3]) < 0.01 and np.linalg.norm(dr[6:9]) < 0.01 : 
            ka = self.ka_near
            self.get_logger().info(f"using ka near (ka = {ka})")
        else:
            ka = self.ka_far
            self.get_logger().info(f"using ka far (ka = {ka})")

        dr = self.vmax * np.tanh(self.k * dr.astype(float).flatten())

        self.get_logger().info(f"dr after saturation: {dr}")

        def decompose_r(r):
            """Decompose the r vector into positions and rotation matrices for left and right end effectors"""
            
            # Extraire les positions
            r_pos = np.array(r[0:3], dtype=np.float64)
            vr = np.array(r[3:6], dtype=np.float64)
            l_pos = np.array(r[6:9], dtype=np.float64)
            vl = np.array(r[9:12], dtype=np.float64)

            return r_pos, l_pos, vr, vl

        # Changement de base: x=0, y=2, z=1
        # Droite
        cmd_right = Twist()
        cmd_right.linear.x = dr[0]
        cmd_right.linear.y = -dr[2]
        cmd_right.linear.z = dr[1]
        cmd_right.angular.x = dr[3] * ka
        cmd_right.angular.y = -dr[5] * ka
        cmd_right.angular.z = dr[4] * ka

        viz_cmd_right = Twist()
        viz_cmd_right.linear.x = dr[0]
        viz_cmd_right.linear.y = dr[2]
        viz_cmd_right.linear.z = dr[1]
        viz_cmd_right.angular.x = dr[3] * ka
        viz_cmd_right.angular.y = dr[5] * ka
        viz_cmd_right.angular.z = dr[4] * ka
        self.cmd_vel_right_pubs.publish(TwistStamped(
            header=Header(stamp=self.get_clock().now().to_msg(),
                   frame_id="fixed_right_gripper_bf"), twist=viz_cmd_right))

        viz_cmd_left = Twist()
        viz_cmd_left.linear.x = dr[6]
        viz_cmd_left.linear.y = dr[8]
        viz_cmd_left.linear.z = dr[7]
        viz_cmd_left.angular.x = dr[9] * ka
        viz_cmd_left.angular.y = dr[11] * ka
        viz_cmd_left.angular.z = dr[10] * ka
        self.cmd_vel_left_pubs.publish(TwistStamped(
            header=Header(stamp=self.get_clock().now().to_msg(),
                   frame_id="fixed_left_gripper_bf"), twist=viz_cmd_left))
        
        # Gauche
        cmd_left = Twist()
        cmd_left.linear.x = dr[6]
        cmd_left.linear.y = -dr[8]
        cmd_left.linear.z = dr[7]
        cmd_left.angular.x = dr[9] * ka
        cmd_left.angular.y = -dr[11] * ka
        cmd_left.angular.z = dr[10] * ka

        self.cmd_vel_left_pub.publish(cmd_left)

        self.cmd_vel_right_pub.publish(cmd_right)
        
        self.get_logger().info(f"norm left linear cmd : {np.linalg.norm([cmd_left.linear.x, cmd_left.linear.y, cmd_left.linear.z])}")
        self.get_logger().info(f"norm right linear cmd : {np.linalg.norm([cmd_right.linear.x, cmd_right.linear.y, cmd_right.linear.z])}")
        self.get_logger().info(f"norm left angular cmd : {np.linalg.norm([cmd_left.angular.x, cmd_left.angular.y, cmd_left.angular.z])}")
        self.get_logger().info(f"norm right angular cmd : {np.linalg.norm([cmd_right.angular.x, cmd_right.angular.y, cmd_right.angular.z])}")
        self.get_logger().info(f"***************")

    def curve_target_callback(self, msg):
        self.s_star = np.array(msg.data).reshape(-1, 1)

        self.get_logger().info(f"Received target shape")

    def cmd_cb(self,msg : Float32MultiArray):

        if self.A is None :
            self.get_logger().info("Waiting for A")
            return
        
        s = np.array(msg.data)        
        Jp = np.linalg.pinv(self.A)

        s = s.reshape(-1,1)
        if self.s_star is None :
            self.get_logger().info(f"Waiting for target shape to pub cmd")
            return

        dsstar = self.s_star - s
        dr_cmd = Jp @ dsstar

        self.pub_cmd(dr_cmd)
        


def main(args=None):
    rclpy.init(args=args)

    point_controller = Broyden_controller_s_Ainit()

    rclpy.spin(point_controller)

    point_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()











        










    




    



