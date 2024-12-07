import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dimensions des segments du bras
L1 = 10  # Longueur du premier segment
L2 = 10  # Longueur du deuxième segment

def rotation_matrix(axis, angle):
    """
    Retourne une matrice de rotation 3D pour une rotation autour d'un axe donné.
    :param axis: Axe de rotation ('x', 'y', 'z')
    :param angle: Angle en radians
    :return: Matrice de rotation 3x3
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axe non valide : choisir parmi 'x', 'y', ou 'z'.")

def inverse_kinematics(x, y, z):
    """
    Calcule les angles des articulations pour atteindre la position cible (x, y, z).
    :param x: Coordonnée x de la cible
    :param y: Coordonnée y de la cible
    :param z: Coordonnée z de la cible
    :return: Angles (phi, theta1, theta2) ou None si la cible est hors de portée
    """
    r = np.sqrt(x**2 + y**2)  # Distance dans le plan (x, y)
    phi = np.arctan2(y, x)  # Angle de rotation de base
    d = np.sqrt(r**2 + z**2)  # Distance totale jusqu'à la cible

    if d > (L1 + L2):
        print("La cible est hors de portée.")
        return None

    cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(cos_theta2)

    cos_theta1 = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    theta1 = np.arctan2(z, r) - np.arccos(cos_theta1)

    return phi, theta1, theta2

def plot_robot_arm(x, y, z):
    """
    Trace le bras robotique en 3D pour atteindre la position cible (x, y, z).
    """
    angles = inverse_kinematics(x, y, z)
    if angles is None:
        return

    phi, theta1, theta2 = angles

    # Matrices de rotation
    R_base = rotation_matrix('z', phi)  # Rotation de la base
    R_joint1 = np.dot(R_base, rotation_matrix('y', theta1))  # Rotation du premier segment
    R_joint2 = np.dot(R_joint1, rotation_matrix('y', theta2))  # Rotation du deuxième segment

    # Positions des articulations
    origin = np.array([0, 0, 0])
    joint1 = origin + np.dot(R_base, np.array([L1, 0, 0]))
    joint2 = joint1 + np.dot(R_joint1, np.array([L2, 0, 0]))

    # Extraction des coordonnées
    x_points = [origin[0], joint1[0], joint2[0]]
    y_points = [origin[1], joint1[1], joint2[1]]
    z_points = [origin[2], joint1[2], joint2[2]]

    # Création du graphique 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracé des segments du bras
    ax.plot(x_points[:2], y_points[:2], z_points[:2], 'r-', linewidth=5, label="Segment 1")
    ax.plot(x_points[1:], y_points[1:], z_points[1:], 'b-', linewidth=5, label="Segment 2")

    # Points des articulations
    ax.scatter(x_points, y_points, z_points, color='k', s=50)
    ax.scatter([x], [y], [z], color='g', s=100, label="Position cible")

    # Réglages de l'affichage
    ax.set_xlim([-L1 - L2, L1 + L2])
    ax.set_ylim([-L1 - L2, L1 + L2])
    ax.set_zlim([0, L1 + L2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Représentation 3D du bras robotique avec matrices de rotation")

    plt.show()

# Saisie des coordonnées cible
x = float(input("Entrez la coordonnée X : "))
y = float(input("Entrez la coordonnée Y : "))
z = float(input("Entrez la coordonnée Z : "))

# Tracé du bras robotique
plot_robot_arm(x, y, z)
