import numpy as np

def distance_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice des distances L2 entre chaque vecteur de X et chaque vecteur de Y.
    
    Paramètres :
    X (np.ndarray) : matrice de taille (n_samples_X, n_features)
    Y (np.ndarray) : matrice de taille (n_samples_Y, n_features)
    
    Retour :
    dists (np.ndarray) : matrice de distances de taille (n_samples_X, n_samples_Y)
                          dists[i,j] = distance L2 entre X[i] et Y[j]
    """
    # Formule vectorisée : ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    X_square = np.sum(X**2, axis=1, keepdims=True)  # (n_samples_X, 1)
    Y_square = np.sum(Y**2, axis=1)  # (n_samples_Y,)
    cross_term = X @ Y.T  # (n_samples_X, n_samples_Y)
    
    dists = np.sqrt(X_square + Y_square - 2 * cross_term)
    return dists


def knn_predict(dists: np.ndarray, labels_train: np.ndarray, k: int) -> np.ndarray:
    """
    Prédit les labels des données test à partir de la matrice de distances et des labels d'entraînement.
    
    Paramètres :
    dists (np.ndarray) : matrice des distances entre train et test (n_test, n_train)
    labels_train (np.ndarray) : labels des données d'entraînement
    k (int) : nombre de voisins
    
    Retour :
    pred_labels (np.ndarray) : labels prédits pour les données test
    """
    n_test = dists.shape[0]
    pred_labels = np.zeros(n_test, dtype=np.int64)
    
    # Sélection rapide des k plus proches voisins
    knn_idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]  # indices des k plus proches

    # Récupérer les labels et voter
    knn_labels = labels_train[knn_idx]  # (n_test, k)
    for i in range(n_test):
        pred_labels[i] = np.bincount(knn_labels[i]).argmax()
    
    return pred_labels


def evaluate_knn(data_train: np.ndarray, labels_train: np.ndarray,
                 data_test: np.ndarray, labels_test: np.ndarray,
                 k: int, batch_size: int = 500) -> float:
    """
    Évalue le classifieur KNN en calculant le taux de bonnes classifications (accuracy).
    Utilise des sous-batchs pour accélérer le calcul et réduire l'utilisation mémoire.

    Paramètres :
    data_train, labels_train : données et labels d'entraînement
    data_test, labels_test : données et labels de test
    k : nombre de voisins
    batch_size : nombre d'exemples test à traiter simultanément

    Retour :
    accuracy (float) : taux de bonnes classifications
    """
    n_test = data_test.shape[0]
    pred_labels = np.zeros(n_test, dtype=np.int64)

    # Boucle sur les sous-batchs de test
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch_test = data_test[start:end]
        # Calculer la matrice des distances pour ce batch uniquement
        dists = distance_matrix(batch_test, data_train)
        # Prédire les labels pour ce batch
        pred_labels[start:end] = knn_predict(dists, labels_train, k)

    # Calculer l'accuracy finale
    accuracy = np.mean(pred_labels == labels_test)
    return accuracy



if __name__ == "__main__":
    # Exemple d'utilisation avec un petit sous-ensemble pour tester
    from read_cifar import read_cifar, split_dataset

    all_data, all_labels = read_cifar("data/cifar-10-batches-py")
    data_train, labels_train, data_test, labels_test = split_dataset(all_data, all_labels, split=0.9)

    # Tester différentes valeurs de k
    import matplotlib.pyplot as plt

    ks = range(1, 21)
    accuracies = []
    for k in ks:
        acc = evaluate_knn(data_train, labels_train, data_test, labels_test, k)
        print(f"k={k}, accuracy={acc:.4f}")
        accuracies.append(acc)

    # Sauvegarder le graphique
    plt.figure()
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel("k (nombre de voisins)")
    plt.ylabel("Accuracy")
    plt.title("KNN sur CIFAR-10")
    plt.grid(True)
    plt.savefig("results/knn.png")
