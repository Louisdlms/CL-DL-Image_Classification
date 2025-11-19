import numpy as np
import pickle

def read_cifar_batch(batch_path: str = "data/cifar-10-batches-py/data_batch_1"):
    # Ouvrir le fichier du batch en mode lecture binaire
    with open(batch_path, 'rb') as f:
        # Charger le contenu avec pickle
        batch = pickle.load(f, encoding='bytes')
    
    # Récupérer les données d'images et les convertir en float32
    data = batch[b'data'].astype(np.float32)
    
    # Récupérer les labels et les convertir en int64
    labels = np.array(batch[b'labels'], dtype=np.int64)
    
    # Retourner les deux tableaux
    return data, labels


def read_cifar(directory_path: str = "data/cifar-10-batches-py"):
    data_list = []
    labels_list = []

    # Lire les 5 batches d'entraînement
    for i in range(1, 6):
        batch_path = f"{directory_path}/data_batch_{i}"
        data, labels = read_cifar_batch(batch_path)
        data_list.append(data)
        labels_list.append(labels)
    
    # Lire le batch de test
    test_path = f"{directory_path}/test_batch"
    data_test, labels_test = read_cifar_batch(test_path)
    data_list.append(data_test)
    labels_list.append(labels_test)
    
    # Combiner toutes les données et labels
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    return all_data, all_labels


def split_dataset(data: np.ndarray, labels: np.ndarray, split: float = 0.8):
    # Mélanger les indices
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    
    # Appliquer le mélange
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]
    
    # Calculer la taille de l'ensemble d'entraînement
    split_idx = int(split * data.shape[0])
    
    # Séparer les ensembles
    data_train = data_shuffled[:split_idx]
    labels_train = labels_shuffled[:split_idx]
    data_test = data_shuffled[split_idx:]
    labels_test = labels_shuffled[split_idx:]
    
    return data_train, labels_train, data_test, labels_test



if __name__ == "__main__":
    # Tester la lecture d'un batch
    batch_path = "data/cifar-10-batches-py/data_batch_1"
    data, labels = read_cifar_batch(batch_path)
    print(f"Lecture d'un batch : data.shape = {data.shape}, data.dtype = {data.dtype}, labels.shape = {labels.shape}, labels.dtype = {labels.dtype}")

    # Tester la lecture de tous les batches (entraînement + test)
    directory_path = "data/cifar-10-batches-py"
    all_data, all_labels = read_cifar(directory_path)
    print(f"Lecture de tous les batches : all_data.shape = {all_data.shape}, all_labels.shape = {all_labels.shape}")

    # Tester la séparation en ensembles d'entraînement et test
    data_train, labels_train, data_test, labels_test = split_dataset(all_data, all_labels, split=0.8)
    print(f"Ensemble d'entraînement : data_train.shape = {data_train.shape}, labels_train.shape = {labels_train.shape}")
    print(f"Ensemble de test : data_test.shape = {data_test.shape}, labels_test.shape = {labels_test.shape}")
