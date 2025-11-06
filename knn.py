import numpy as np
import matplotlib.pyplot as plt
import read_cifar as rc


def distance_matrix(m1:np.ndarray, m2:np.ndarray) :
    # Normes au carré
    n1 = np.sum(m1 * m1, axis=1)            # (N,)
    n2 = np.sum(m2 * m2, axis=1)            # (M,)
    # Distances au carré via (a-b)^2 = a^2 + b^2 - 2ab
    d2 = n1[:, None] + n2[None, :] - 2.0 * (m1 @ m2.T)
    # Garde numérique: clamp les petites valeurs négatives à 0
    dists = np.maximum(d2, 0.0, out=d2)
    return dists


def knn_predict(dists, labels_train, k):
    # Indices des k plus proches voisins pour chaque test (ordre croissant de distance)
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]  # plus rapide que argsort complet
    # Pour être sûr d’un tri total des k plus proches (optionnel):
    # on trie localement ces k indices selon la distance réelle
    row_idx = np.arange(dists.shape[0])[:, None]
    local_order = np.argsort(dists[row_idx, idx], axis=1)
    knn_idx = idx[row_idx, local_order]  # (N_test, k)

    # Récupération des labels des k voisins
    knn_labels = labels_train[knn_idx]  # (N_test, k)

    # Vote majoritaire par ligne via np.bincount
    # On applique ligne par ligne, np.bincount nécessite des entiers >= 0
    preds = np.empty(knn_labels.shape[0], dtype=labels_train.dtype)
    for i in range(knn_labels.shape[0]):
        counts = np.bincount(knn_labels[i])
        preds[i] = counts.argmax()
    return preds

def evaluate_knn(data_train, labels_train, data_test, labels_test, k, batch_size=128):
    N_test = data_test.shape[0]
    preds_parts = []
    for start in range(0, N_test, batch_size):
        stop = start + batch_size
        dists = distance_matrix(data_test[start:stop], data_train)
        preds = knn_predict(dists, labels_train, k)
        preds_parts.append(preds)
    preds_all = np.concatenate(preds_parts, axis=0)
    return float(np.mean(preds_all == labels_test))

if __name__ == "__main__":
    data, labels = rc.read_cifar('data/cifar-10-batches-py')
    data_train, labels_train, data_test, labels_test = rc.split_dataset(data, labels, 0.9)

    ks = list(range(1, 21))
    accuracies = []
    for k in ks:
        acc = evaluate_knn(data_train, labels_train, data_test, labels_test, k, batch_size=256)
        print(f"k={k}: accuracy={acc:.4f}")
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('k-NN accuracy vs k (split=0.9)')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('Plots/knn_accuracy.png')