import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray,Float32MultiArray

from geometry_msgs.msg import Point, Twist,TwistStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped


M = 3 # degree of polynomial

P = 40

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    r_pos, l_pos, r_R, l_R = [], [], [], []
    points_between_tcp = []
    curves = []

    i=0

    for line in lines:
        line = line.strip()
        if line.startswith("Right TCP:"):
            i+=1

            if i%P == 0 :r_pos.append(np.array([float(x) for x in line.split(":")[1].split(",")]))

        elif line.startswith("Left TCP:"):
            if i%P == 0 :l_pos.append(np.array([float(x) for x in line.split(":")[1].split(",")]))

            if i%P == 0 :curves.append(points_between_tcp)
            points_between_tcp = []  # Reset for the next curve

        elif line.startswith("Right Orientation:"):
            r_quat = [float(x) for x in line.split(":")[1].split(",")]
            if i%P == 0 :r_R.append(R.from_quat(r_quat).as_matrix())

        elif line.startswith("Left Orientation:"):
            l_quat = [float(x) for x in line.split(":")[1].split(",")]
            if i%P == 0 :l_R.append(R.from_quat(l_quat).as_matrix())
        elif line.startswith("(") and line.endswith(")"):
            point = [float(x) for x in line.strip("()").split(",")]
            if i%P == 0 :points_between_tcp.append(point)

    curves = np.array(curves)

    return r_pos, l_pos, r_R, l_R, curves


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

def compute_r_from_transes(tr :TransformStamped,tl :TransformStamped):
    """Compute the r vector from the transforms of the left and right end effectors"""
    r_left, pos_left = trans_to_vecs(tl)
    r_right, pos_right = trans_to_vecs(tr)

    r = np.array([pos_right[0],pos_right[1],pos_right[2],
                  r_right[0],r_right[1],r_right[2],
                  pos_left[0],pos_left[1],pos_left[2],
                  r_left[0],r_left[1],r_left[2]], dtype=np.float64)
    
    return r

def compute_r(r_pos, l_pos, r_R, l_R):
    """Compute the r vector from the positions and orientations of the left and right end effectors"""

    vr = R.from_matrix(r_R).as_rotvec()
    vl = R.from_matrix(l_R).as_rotvec()
    r = np.array([r_pos[0], r_pos[1], r_pos[2],
                vr[0], vr[1], vr[2],
                l_pos[0], l_pos[1], l_pos[2],
                vl[0], vl[1], vl[2]]
                , dtype=np.float64)
    
    return r

from scipy.spatial.transform import Rotation as R
import numpy as np

def decompose_r(r):
    """Decompose the r vector into positions and rotation matrices for left and right end effectors"""
    
    # Extraire les positions
    r_pos = np.array(r[0:3], dtype=np.float64)
    vr = np.array(r[3:6], dtype=np.float64)
    l_pos = np.array(r[6:9], dtype=np.float64)
    vl = np.array(r[9:12], dtype=np.float64)
    
    # Convertir les vecteurs de rotation en matrices de rotation
    r_R = R.from_rotvec(vr.transpose()).as_matrix()
    l_R = R.from_rotvec(vl.transpose()).as_matrix()
    
    return r_pos, l_pos, vr, vl


def computeWi(r):
    """r : iterable of scalar dof of effectors"""
    w = np.matrix([[]])
    for ri in r:
        w = np.hstack([w,np.matrix([[ ri**j for j in range(M) ]])])
    zero = np.zeros_like(w)
    W = np.vstack([np.hstack([w,zero,zero]),np.hstack([zero,w,zero]),np.hstack([zero,zero,w])])    
    
    return W

def compute_W(r,n):
    Wi = computeWi(r)
    W = np.kron(np.eye(n), Wi)
    return W

def compute_dWi(r):
    """r : iterable of scalar dof of effectors"""

    dW = np.matrix([[]])

    for ri in r:

        dWdri = np.matrix([[j*ri**(j-1) for j in range(M)]])
        dW = np.hstack([dW, dWdri])

    return dW

    


def compute_A(beta,r,n=51):

    R = 12
    dwi = compute_dWi(r)
    j=0

    A = np.vstack( 
        np.hstack(
            [dwi[:,3*i : 3*(i+1)] @ beta[3*i + R*j:3*(i+1) + R*j]  for i in range(R)]
            ) for j in range(3*n))
    
    print(f"A shape: {A.shape}")

    return np.asarray(A)

    


def compute_cmd(beta,s,s_star,r):

    A = compute_A(beta,r,n=s.shape[0]/3)

    ds = s_star - s


    invA = np.linalg.pinv(A)

    dr = invA @ ds

    dr_pos, dl_pos, dvr, dvl = decompose_r(dr.transpose())




class Ainit_computer(Node):
    def __init__(self):
        super().__init__('a_init_computer')




        self.tcp_left = None
        self.tcp_right = None


        self.A = None

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

        self.beta_sub = self.create_subscription(
            Float32MultiArray,  
            '/beta',
            self.beta_callback,
            10
        )

        self.A_pub = self.create_publisher(Float64MultiArray, '/A_init', 10)

        self.beta = None

        self.create_timer(0.1, self.timer_cb)
        


    def tcp_right_callback(self, msg):
        self.get_logger().info(
            f"tcp_right: position=({msg.transform.translation.x}, {msg.transform.translation.y}, {msg.transform.translation.z}), "
            f"orientation=({msg.transform.rotation.x}, {msg.transform.rotation.y}, {msg.transform.rotation.z}, {msg.transform.rotation.w})"
        )
        self.tcp_right = msg

    def tcp_left_callback(self, msg):
        self.get_logger().info(
            f"tcp_left: position=({msg.transform.translation.x}, {msg.transform.translation.y}, {msg.transform.translation.z}), "
            f"orientation=({msg.transform.rotation.x}, {msg.transform.rotation.y}, {msg.transform.rotation.z}, {msg.transform.rotation.w})"
        )
        self.tcp_left = msg


    def timer_cb(self):
        if self.tcp_left is not None and self.tcp_right is not None and self.beta is not None:

            self.get_logger().info("Computing A matrix...")
            r = compute_r_from_transes(self.tcp_right, self.tcp_left)
            A = compute_A(self.beta, r, n=51)

            self.get_logger().info(f"A matrix computed, shape: {A.shape}")
            A_msg = Float64MultiArray()

            A_list = A.flatten().tolist()

            self.get_logger().info(f"Publishing A matrix with {len(A_msg.data)} elements. A0 = {A_list[0]}")

            A_msg.data = A_list


            self.A_pub.publish(A_msg)


    def beta_callback(self, msg):
        self.get_logger().info(f"Received beta with {len(msg.data)} elements.")
        self.beta = beta = np.array(msg.data).reshape(-1, 1)




    

def main(args=None):
    rclpy.init(args=args)
    node = Ainit_computer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()