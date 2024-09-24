import numpy as np
import trimesh
import scipy.spatial
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree

# Função para carregar a nuvem de pontos de um arquivo que está dentro de uma pasta com o nome numérico
def load_point_cloud(file_name, downsample_factor=10):
    # Extrair a parte numérica do nome do arquivo (antes do underline)
    base_name = file_name.split('_')[0]  # Pega apenas a parte numérica antes do primeiro underline

    # Construir o caminho da pasta que tem o nome numérico
    file_path = os.path.join("/content/drive/MyDrive/KITTI-Sequence", base_name, file_name)

    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        raise ValueError(f"Arquivo não encontrado: {file_path}")

    # Carregar a nuvem de pontos e acessar os vértices diretamente
    point_cloud = trimesh.load(file_path).vertices

    # Downsample (reduzir a resolução) da nuvem de pontos
    if downsample_factor > 1:
        point_cloud = point_cloud[::downsample_factor]

    return point_cloud

# Função para calcular a melhor transformação entre dois conjuntos de pontos
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # Centroid of each point cloud
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Covariance matrix
    H = np.dot(AA.T, BB)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t

# Função para filtrar outliers com base na distância mediana
def filter_outliers(A, B, distances, threshold=1.5):
    median_distance = np.median(distances)
    valid_mask = distances < (median_distance * threshold)
    return A[valid_mask], B[valid_mask]

# Função do ICP usando KDTree, best_fit_transform e filtragem de outliers
def icp(A, B, max_iterations=50, tolerance=1e-6, outlier_threshold=1.5):
    src = np.copy(A)
    dst = np.copy(B)

    prev_error = float('inf')

    for i in range(max_iterations):
        # Encontrar os vizinhos mais próximos
        tree = KDTree(dst)
        distances, indices = tree.query(src)

        # Filtrar outliers
        src_filtered, dst_filtered = filter_outliers(src, dst[indices], distances, outlier_threshold)

        # Computar a transformação rígida
        R, t = best_fit_transform(src_filtered, dst_filtered)

        # Transformar os pontos de origem
        src = np.dot(src, R.T) + t

        # Calcular o erro médio
        mean_error = np.mean(distances)

        # Verificar convergência
        if np.abs(prev_error - mean_error) < tolerance:
            print(f"ICP convergiu em {i+1} iterações.")
            break
        prev_error = mean_error

    return R, t, mean_error

# Função para acumular transformações
def accumulate_transformation(T_prev, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return np.dot(T_prev, T)

# Função para calcular a trajetória carregando os scans em lotes
def calculate_trajectory_in_batches(n_scans, downsample_factor=2):
    trajectory = np.zeros((n_scans, 3))
    cumulative_transform = np.eye(4)  # Inicialize com a matriz identidade

    for i in range(1, n_scans):
        # Carregar as nuvens de pontos
        src_points = load_point_cloud(f"{i-1:06}_points.obj", downsample_factor=downsample_factor)
        tgt_points = load_point_cloud(f"{i:06}_points.obj", downsample_factor=downsample_factor)

        # Executar o ICP com filtragem de outliers
        R, t, _ = icp(src_points, tgt_points, max_iterations=100, outlier_threshold=1.5)

        # Atualizar a transformação cumulativa
        cumulative_transform = accumulate_transformation(cumulative_transform, R, t)

        # Armazenar a posição atual do carro na trajetória
        trajectory[i] = cumulative_transform[:3, 3]

    return trajectory

# Carregar ground-truth
ground_truth = np.load("/content/drive/MyDrive/ground_truth/ground_truth.npy")

# Estimar a trajetória usando ICP em lotes com downsampling ajustável
estimated_trajectory = calculate_trajectory_in_batches(30, downsample_factor=1)  # Usar downsample menor

# Comparar com a ground-truth e imprimir o erro
error = np.mean(np.linalg.norm(estimated_trajectory - ground_truth[:, :3, 3], axis=1))
print(f"Erro médio da trajetória estimada: {error:.4f}")

# Plotar a trajetória estimada em comparação com a ground-truth
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], label='Trajetória Estimada')
ax.plot(ground_truth[:, 0, 3], ground_truth[:, 1, 3], label='Ground-Truth', linestyle='dashed')
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.legend()
plt.show()
