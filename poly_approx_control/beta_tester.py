import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

M = 3 # degree of polynomial

P = 320



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

    


def compute_A(r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes,r):
    N = len(r_poses)
    n = len(points_between_tcpes[0])
    R = 12

    print(f"Number of curves: {N}, Number of points per curve: {n}")

    print(f"W shape: {compute_W(r, n).shape}")

    sv = np.array(points_between_tcpes).flatten()

    Wv = np.vstack([compute_W(compute_r(r_poses[i], l_poses[i], r_Rs[i], l_Rs[i]), n) for i in range(N)])

    invWv = np.linalg.pinv(Wv)
    beta = (invWv @ sv).transpose()

    print(f"beta shape: {beta.shape}")
    print(f"Wv shape: {Wv.shape}")
    print(f"sv shape: {sv.shape}")

    print(f"r : {r}")
    print(f"r shape: {r.shape}")

    print(f"dwi  : {compute_dWi(r).shape}")

    dwi = compute_dWi(r)

    r_test = [2,10,20]

    j=0

    a= np.hstack(
            [dwi[:,3*i : 3*(i+1)] @ beta[3*i + M*j:3*(i+1) + R*j]  for i in range(R)]
            )
    
    print(f"a shape: {a.shape}")
    



    A = np.vstack( 
        np.hstack(
            [dwi[:,3*i : 3*(i+1)] @ beta[3*i + R*j:3*(i+1) + R*j]  for i in range(R)]
            ) for j in range(3*n))
    
    print(f"A shape: {A.shape}")

    return A

def plot_cable(s,ax,color='b'):
    """
    Plots a 3D cable given its points.
    
    Parameters:
        s (numpy.ndarray): A (51, 3) array representing the cable points (x, y, z).
    """
    if s.shape != (51, 3):
        s = s.reshape(-1, 3)  # Ensure s is reshaped to (51, 3)
    

    
    ax.plot(s[:, 0], s[:, 1], s[:, 2], marker='o', label='Cable', color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cable Plot')
    ax.legend()
    



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

class BetaTesterNode(Node):
    def __init__(self):
        super().__init__('beta_tester')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/beta',
            self.beta_callback,
            10
        )
        self.get_logger().info("beta_tester node is running and listening to /beta topic.")

    def beta_callback(self, msg):

        beta = np.array(msg.data)

        r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes =read_txt_file("curve_points_datas.txt")

        r = compute_r(r_poses[0], l_poses[0], r_Rs[0], l_Rs[0])

        W = compute_W(r, len(points_between_tcpes[0]))

        print(f"Computed W shape: {W.shape}")
        print(f"Received beta shape: {beta.shape}")

        s = np.asarray(W @ beta)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot_cable(s,ax)

        plt.show()













def main(args=None):
    rclpy.init(args=args)
    node = BetaTesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()