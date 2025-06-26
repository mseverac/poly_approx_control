from geometry_msgs.msg import Point,Pose,PoseArray,Twist,Vector3,TransformStamped
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension,Float32MultiArray
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

def broyden_update(A,ds,dr,gamma=0.01):
    """Update the Jacobian matrix A using Broyden's method"""

    dr = np.asarray(dr).reshape(-1,1)
    ds = np.asarray(ds).reshape(-1,1)

    print(f" A shape: {A.shape}, ds shape: {ds.shape}, dr shape: {dr.shape}")
    print(f"A @ dr shape: {(np.asarray(A @ dr).reshape(-1,1)).shape}, ds shape: {ds.shape}")
    print(f"dr.T @ dr shape: {(dr.T @ dr).shape}")
    print(f"ds - A @ dr shape: {(ds - np.asarray(A @ dr).reshape(-1,1)).shape}")
    print(f"dr.T shape: {dr.T.shape}")
    print(f"  ((ds - A @ dr) / dr.T @ dr ) shape: {((ds - A @ dr) / (dr.T @ dr)).shape}")
    A = A + gamma * ((ds - np.asarray(A @ dr).reshape(-1,1)) / dr.T @ dr ) @ dr.T


    return A

M=4 # degree of polynomial
d=2 # dof of each point
k_angles = 100 #gain pour la commande des angles
file_path = "curve_poses_log.txt"



class PointController(Node):
    def __init__(self):
        super().__init__('point_controller')

        self.tcp_left,self.tcp_right = None, None

        self.publisher_r = self.create_publisher(Twist, '/right/vis_vel_cmd_6dof', 10)
        self.publisher_l = self.create_publisher(Twist, '/left/vis_vel_cmd_6dof', 10)

         
        self.create_subscription(
            Float32MultiArray,
            'points3d',
            self.points3d_callback,
            10
        )
        
        self.create_subscription(Float32MultiArray,"/curve_target_6dof",self.curve_target_callback,1)

        self.create_timer(0.1, self.timer_cb)

        self.create_timer(1.0, self.broyden_cb)
        
        
        self.curve_target_data = None

        self.point_index = 5

        self.ka = 0.03
        self.k = 2.0
        self.vmax = 0.02

        self.init_cmd = np.random.rand(12) * 0.1


        self.prev_error = np.zeros(6)
        self.integral_error = np.zeros(6)
        self.last_time = self.get_clock().now()



        self.tcp_left = None
        self.tcp_right = None

        self.points3d_data = None
        self.last_points3d_time = self.get_clock().now()

        self.dr = None
        self.ds = None
        self.last_s = None
        self.last_r = None

        self.A = None
        self.gamma = 0.01

        self.cmd_vel_left_pub = self.create_publisher(
            Twist, "/left/vis_vel_cmd_6dof", 1
        )
        self.cmd_vel_right_pub = self.create_publisher(
            Twist, "/right/vis_vel_cmd_6dof", 1
        )


                
                
        self.tcp_left_sub = self.create_subscription(
            TransformStamped,
            '/tcp_left',
            self.tcp_left_callback,
            10
        )
        self.tcp_right_sub = self.create_subscription(
            TransformStamped,
            '/tcp_right',
            self.tcp_right_callback,
            10
        )
        


    def tcp_right_callback(self, msg):
        
        self.tcp_right = msg

    def tcp_left_callback(self, msg):
        
        self.tcp_left = msg

    def broyden_cb(self):
        missing_params = []
        if self.points3d_data is None:
            missing_params.append("points3d_data")
        if self.tcp_left is None:
            missing_params.append("tcp_left")
        if self.tcp_right is None:
            missing_params.append("tcp_right")


        if missing_params:
            self.get_logger().info(f"missing : {', '.join(missing_params)} for Broyden update")
                
            time.sleep(0.2)  # Sleep to avoid busy waiting

        else:

            if self.last_s is None:
                self.last_s = self.points3d_data
                self.last_r = compute_r(self.tcp_right, self.tcp_left)

                self.get_logger().info(f"Initializing last_s and last_r and publishing initial command")
                self.pub_cmd_nn(self.init_cmd)
                return
            
            ds = self.points3d_data - self.last_s
            dr = compute_r(self.tcp_right, self.tcp_left) - self.last_r
            norm_ds = np.linalg.norm(ds)
            norm_dr = np.linalg.norm(dr)

            if norm_dr < 0.01 or norm_ds < 0.01:
                self.get_logger().info(f"norm ds or dr too small, skipping broyden update")

                if self.A is None :
                    self.get_logger().info(f"A is None,pubing initial command")
                    self.pub_cmd_nn(self.init_cmd)
                return
            
            if self.A is None:
                self.get_logger().info(f"A is None, initializing A")
                self.A = np.zeros((153, 12))  
                self.A = broyden_update(self.A, ds, dr, gamma=1)

            else:
                self.get_logger().info(f"Updating A with broyden update")
                self.A = broyden_update(self.A, ds, dr, gamma=self.gamma)



    def points3d_callback(self, msg):

        self.points3d_data = np.array(msg.data).reshape(-1, 1)
        self.s = np.array(msg.data).reshape(-1, 1)

        #self.get_logger().info(f"Points3D data shape : {self.points3d_data.shape}")

        self.last_points3d_time = self.get_clock().now()
        
    def curve_target_callback(self, msg):
        self.curve_target_data = np.array(msg.data).reshape(-1, 1)

        self.get_logger().info(f"Received target shape")

    def pub_cmd_nn(self,dr):
        """Publishes the command velocities to the robot for the nn dr"""

        self.get_logger().info(f"------ Command velocities ---")

        # dr = [linear_r(0:3), angular_r(3:6), linear_l(6:9), angular_l(9:12)]
        self.get_logger().info(f"norm right linear dr : {np.linalg.norm(dr[0:3])}")
        self.get_logger().info(f"norm right angular dr : {np.linalg.norm(dr[3:6])}")
        self.get_logger().info(f"norm left linear dr : {np.linalg.norm(dr[6:9])}")
        self.get_logger().info(f"norm left angular dr : {np.linalg.norm(dr[9:12])}")

        self.get_logger().info(f"***************")

        dr = self.vmax * np.tanh(self.k * dr.astype(float).flatten())

        # Changement de base: x=0, y=2, z=1
        # Droite
        cmd_right = Twist()
        cmd_right.linear.x = dr[0]
        cmd_right.linear.y = -dr[2]
        cmd_right.linear.z = dr[1]
        cmd_right.angular.x = dr[3] * self.ka
        cmd_right.angular.y = -dr[5] * self.ka
        cmd_right.angular.z = dr[4] * self.ka

        # Gauche
        cmd_left = Twist()
        cmd_left.linear.x = dr[6]
        cmd_left.linear.y = -dr[8]
        cmd_left.linear.z = dr[7]
        cmd_left.angular.x = dr[9] * self.ka
        cmd_left.angular.y = -dr[11] * self.ka
        cmd_left.angular.z = dr[10] * self.ka

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


    def timer_cb(self):
        missing_params = []
        if self.points3d_data is None:
            missing_params.append("points3d_data")
        if self.curve_target_data is None:
            missing_params.append("curve_target_data")
        if self.tcp_left is None:
            missing_params.append("tcp_left")
        if self.tcp_right is None:
            missing_params.append("tcp_right")
        if self.A is None:
            missing_params.append("A")

        if missing_params:
            self.get_logger().info(f"missing : {', '.join(missing_params)} for computing cmd")
            time.sleep(0.2)  # Sleep to avoid busy waiting

        else:

            s = self.points3d_data.copy()
            sstar = self.curve_target_data.copy()
            ds_cmd = sstar - s
            #self.get_logger().info(f"ds shape: {ds.shape}")

            invJ = np.linalg.pinv(self.A)
            #self.get_logger().info(f"invJ shape: {invJ.shape}")

            dr_cmd = invJ @ ds_cmd

            self.pub_cmd_nn(dr_cmd)



def main(args=None):
    rclpy.init(args=args)

    point_controller = PointController()

    rclpy.spin(point_controller)

    point_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()











        










    




    



