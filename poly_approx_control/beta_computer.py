import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension
import numpy as np



M=4 # degree of polynomial
d=2 # dof of each point
k_angles = 100 #gain pour la commande des angles
file_path = "curve_poses_log.txt"



def read_curve_points(file_path):
    data = []
    current_curve = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '---':  # Separator between curves
                if current_curve:
                    data.append(np.array(current_curve))
                    current_curve = []
            elif line:  # Non-empty line
                values = list(map(float, line.split(',')))
                if len(values) == 4:  
                    x,y,z,w = values
                    values = Rotation.quaternion(x,y,z,w).get_rpy()
                    values = k_angles*np.array(values,dtype=np.float64)
                current_curve.append(values)
        if current_curve:  # Add the last curve if it exists
            data.append(np.array(current_curve))

    N = len(data)
    return np.array(data, dtype=object)  # Use dtype=object for jagged arrays





def computeWi(r):
    """r : iterable of scalar dof of effectors"""
    w = np.matrix([[]])
    for ri in r:
        w = np.hstack([w,np.matrix([[ ri**j for j in range(M) ]])])
    zero = np.zeros_like(w)
    W = np.vstack([np.hstack([w,zero]),np.hstack([zero,w])])    
    
    return W

def compute_dWi(r):
    """r : iterable of scalar dof of effectors"""

    dW = np.matrix([[]])

    for ri in r:

        dWdri = np.matrix([[j*ri**(j-1) for j in range(M)]])
        dW = np.hstack([dW, dWdri])

    return dW

def computeA(r, beta, i):
    beta_i = beta[i]
    m = len(r)

    A = None

    for k in range(d):
        a = np.matrix([[]], dtype=np.float64)
        for l in range(m):
            a = np.hstack([a, np.matrix([sum([j * beta_i[j + l * M + m * k * M] * r[l] ** (j - 1) for j in range(M)])], dtype=np.float64)])
        if A is None:
            A = a
        else:
            A = np.vstack([A, a])

    return A


def computeBetai(curve_points,j):

    s1v = np.array([curve_points[i,j,:2] for i in range(len(curve_points))]).flatten()
    W1v = np.vstack([computeWi(compute_r(curve)) for curve in curve_points]) 

    invW1v = np.linalg.pinv(W1v)
    betai = (invW1v @ s1v).transpose()

    return betai



def compute_r(curve):
    r = np.array([curve[0,0],curve[0,1],curve[1,1],curve[-2,0],curve[-2,1],curve[-1,1]]).flatten()
    return r

class BetaComputer(Node):
    def __init__(self):
        super().__init__('beta_computer')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/beta', 10)
        self.timer = self.create_timer(0.1, self.compute_and_publish_beta)  # 10 Hz
        self.timer2 = self.create_timer(0.1, self.publish_beta)
        self.beta = None 
        self.get_logger().info('BetaComputer node has been started.')

    def publish_beta(self,beta=None):
        if self.beta is not None :
            beta = self.beta
            msg = Float64MultiArray()
            msg.data = beta.flatten().tolist()  # Convertir en liste pour le message

            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim[0].label = "dim0"
            msg.layout.dim[0].size = 18
            msg.layout.dim[0].stride = 48 * 1  # size of next dimension

            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim[1].label = "dim1"
            msg.layout.dim[1].size = 48
            msg.layout.dim[1].stride = 1 * 1  # size of next dimension

            msg.layout.dim.append(MultiArrayDimension())
            msg.layout.dim[2].label = "dim2"
            msg.layout.dim[2].size = 1
            msg.layout.dim[2].stride = 1  # innermost

            msg.layout.data_offset = 0
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published beta: {beta}')


    def compute_and_publish_beta(self):
        # Exemple de calcul de beta (à adapter selon vos besoins)

        curve_points = read_curve_points(file_path)
        beta = np.array([computeBetai(curve_points,j) for j in range(2,20)])
        self.beta = beta 
        
        # Création et publication du message
        self.publish_beta()
        

    

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