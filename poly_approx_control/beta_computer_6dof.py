import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension,Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


M = 3 # degree of polynomial

P = 10

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
    r_left, pos_left = trans_to_matrix(tl)
    r_right, pos_right = trans_to_matrix(tr)

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

    


def compute_beta(r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes):
    N = len(r_poses)
    n = len(points_between_tcpes[0])
    R = 12

    print(f"Number of curves: {N}, Number of points per curve: {n}")

    sv = np.array(points_between_tcpes).flatten()

    #print(f"sv : {sv}")

    Wv = np.vstack([compute_W(compute_r(r_poses[i], l_poses[i], r_Rs[i], l_Rs[i]), n) for i in range(N)])

    invWv = np.linalg.pinv(Wv)
    beta = (invWv @ sv).transpose()

    print("f beta shape {beta.shape}")

    return np.asarray(beta )


class BetaComputer(Node):
    def __init__(self):
        super().__init__('beta_computer')
        self.beta_pub = self.create_publisher(Float32MultiArray, '/beta', 10)


        

        r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes =read_txt_file("curve_points_datas.txt")

        self.get_logger().info(f"Number of curves: {len(r_poses)}, Number of points per curve: {len(points_between_tcpes[0])}")

        self.get_logger().info(f"begining to compute beta")

        self.beta = compute_beta(r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes)

        self.get_logger().info(f"beta computed, shape: {self.beta.shape} , starting publishing")
        self.create_timer(0.1, self.publish_beta)  


    def publish_beta(self):
        beta_msg = Float32MultiArray()
        beta_list = self.beta.ravel().tolist()
        beta_msg.data = beta_list

        self.beta_pub.publish(beta_msg)
        self.get_logger().info(f"Published beta with {len(beta_msg.data)} elements.")
        

    

def main(args=None):
    rclpy.init(args=args)
    node = BetaComputer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()