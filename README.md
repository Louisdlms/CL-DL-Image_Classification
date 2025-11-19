# CL-DL-Image_Classification

Ce projet vise à classifier les images issues de la base de données CIFAR-10 avec l'implémentation des algorithmes KNN et MLP via NumPy.

## Description

Ce dépôt contient des implémentations de deux approches de classification : un classifieur K-Nearest Neighbors et un réseau de neurones simple (MLP).

Ces 2 implémentations sont réalisées et évaluées à l'aide de la bibliothèque NumPy.

## Structure du dépôt

```
.
├── mlp.py                          # Implémentation MLP
├── knn.py                          # Implémentation KNN
├── read_cifar.py                   # Utilitaires de chargement de données
├── requirements.txt                # Dépendances
├── README.md                       # Ce fichier
├── data/
│   └── cifar-10-batches-py/        # Données CIFAR-10
└──results/                         # Graphiques de sortie
    ├── knn.png
    └── mlp.png
```

## Installation

**requirements** :
```bash
pip install -r requirements.txt
```

## Fichiers et explications techniques

### `read_cifar.py`

Ce fichier contient les utilitaires pour charger et prétraiter les données CIFAR-10.

**Fonctions principales** :

- **`read_cifar_batch(batch_path)`** : 
  - Lit un fichier batch CIFAR-10 à l'aide de `pickle`.
   - Convertit les données en `float32` et labels en `int64`.
  - Retourne la matrice des images (taille : `(batch_size, 3072)`).
 

- **`read_cifar(directory_path)`** : 
  - Boucle sur les 5 fichiers d'entraînement (`data_batch_1..5`) et le fichier de test (`test_batch`).
  - Concatène toutes les données en un seul tableau NumPy.
  - Retourne : données complètes (~60 000 images) et labels correspondants.

- **`split_dataset(data, labels, split=0.8)`** :
  - Mélange aléatoirement les indices des données.
  - Sépare en ensemble d'entraînement (80% par défaut) et ensemble de test (20%).
  - Retourne : `data_train`, `labels_train`, `data_test`, `labels_test`.



### `knn.py`

Implémentation vectorisée d'un classifieur K-Nearest Neighbors pour la classification multi-classe.

**Fonctions principales** :

- **`distance_matrix(X, Y)`** :
  - Calcule la matrice des distances L2 entre tous les points de `X` et tous ceux de `Y`.
  - Utilise la formule vectorisée : $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2 a \cdot b$.
  - Retourne : matrice de taille `(n_samples_X, n_samples_Y)`.
  - implémentation vectorisée avec NumPy (pas de boucle for).

- **`knn_predict(dists, labels_train, k)`** :
  - Sélectionne les `k` plus proches voisins pour chaque exemple de test.
  - Utilise `np.argpartition` pour trouver rapidement les `k` plus petites distances.
  - Effectue un vote majoritaire parmi les `k` labels voisins.
  - Retourne : tableau des labels prédits.

- **`evaluate_knn(data_train, labels_train, data_test, labels_test, k, batch_size=500(optionnel))`** :
  - Évalue l'accuracy du KNN sur l'ensemble de test.
  - Traite les données par batch pour limiter la consommation mémoire (optionnel ici vis à vis de l'énoncé mais plus performant).
  - Retourne : accuracy (proportion de prédictions correctes).

**Script principal** :
- Charge CIFAR-10 et le split train/test.
- Teste différentes valeurs de `k` (de 1 à 20).
- Génère `results/knn.png` : courbe d'accuracy en fonction de `k`.



### `mlp.py`

Implémentation d'un perceptron multi-couches (MLP) à une couche cachée, entraîné par rétropropagation avec descente de gradient.

**Fonctions principales** :

- **`sigmoid(x)` et `sigmoid_derivative(a)`** :
  - Fonction d'activation sigmoïde : $\sigma(x) = \frac{1}{1 + e^{-x}}$.
  - Dérivée : $\sigma'(a) = a(1-a)$.

- **`one_hot(labels)`** :
  - Convertit un vecteur de labels en matrice one-hot.
  - Exemple : label `[1, 2, 0]` → matrice `(3, 3)` avec 1 dans les positions correspondantes.

- **`learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate)`** :
  - Effectue une étape de gradient descendant avec **MSE (Mean Squared Error)** comme fonction coût.
  - Forward pass
  - Backward pass
  - Retourne : les poids/biais mis à jour et la valeur de la MSE.

- **`learn_once_cross_entropy(w1, b1, w2, b2, data, targets, learning_rate)`** :
  - Étape de gradient avec **cross-entropy** pour la classification multi-classe.
  - Forward pass
  - Softmax : $a_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.
  - Loss cross-entropy : $-\sum_i y_i \log(a_i)$.
  - Backward pass : gradients calculés en tenant compte du softmax.
  - Retourne : les poids/biais mis à jour et la valeur de la cross-entropy.

- **`train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)`** :
  - Entraîne le MLP pendant `num_epoch` epochs complets.
  - **Mini-batching** : traite les données par chunks de 64 exemples (Cette étape n'était pas demandée mais permet d'obtenir de meilleurs résultats : on passe de ~18% à ~45% d'accuracy avec une taille de 64 en mini-batch).
  - Tous les 10 epochs, on affiche la loss et l'accuracy d'entraînement.
  - Retourne : les poids/biais finaux et l'historique des accuracies.

- **`test_mlp(w1, b1, w2, b2, data_test, labels_test)`** :
  - Évalue le réseau entraîné sur les données de test.
  - Forward pass uniquement.
  - Retourne : l'accuracy sur les données de test.

- **`run_mlp_training(data_train, labels_train, data_test, labels_test, d_h=64, learning_rate=0.1, num_epoch=100)`** :
  - réalise les 3 étapes suivantes : initialisation, entraînement, évaluation.
  - Initialisation aléatoire : $w \sim U(-1, 1)$, $b = 0$.
  - Normalisation : données divisées par 255 (correspondant à la valeur max pour un pixel donné).
  - Retourne : la liste des accuracies d'entraînement et l'accuracy finale sur test.

**Script principal** :
- Effectue plusieurs tests unitaires (forward pass, loss, gradients).
- Charge CIFAR-10, entraîne un MLP avec les hyperparamètres définis.
- Affiche l'accuracy finale sur le jeu de test.
- Génère `results/mlp.png` : courbe d'évolution de l'accuracy d'entraînement par epoch.

**Hyperparamètres modifiables** (dans le bloc `if __name__ == "__main__"`) :
- `d_h` : nombre de neurons dans la couche cachée (par défaut 64).
- `learning_rate` : taux d'apprentissage (par défaut 0.1).
- `num_epoch` : nombre d'epochs d'entraînement (par défaut 100).
- `batch_size` : taille des mini-batchs (par défaut 64, dans `train_mlp`).

**Résultats attendus** :
- L'accuracy d'entraînement augmente généralement au fil des epochs.
- L'accuracy de test converge mais peut être inférieure à celle d'entraînement (overfitting possible).
- Performances typiques : ~45% sur CIFAR-10 (réseau simple sans augmentation de données ni CNN).


## Usage et interprétation

1. **Tester l'extraction des données** :
   ```bash
   python3 read_cifar.py
   ```
   Affiche les formes des données.

2. **Tester KNN** :
   ```bash
   python3 knn.py
   ```
   Génère `results/knn.png`. Observez comment l'accuracy change avec `k`.

**Résultats observés** :
- Sur CIFAR-10 avec pixels bruts, les performances restent modestes (~30-35%) car KNN est assez peu adapté aux traitements d'images.

3. **Tester MLP** :
   ```bash
   python3 mlp.py
   ```
   Génère `results/mlp.png`. Observez la courbe d'entraînement et l'accuracy finale.

**Résultats observés** :
- L'accuracy d'entraînement augmente au fil des epochs et semble converger.
- L'accuracy de test est légèrement plus faible que celle d'entrainement mais néanmoins très proche donc pas d'overfitting.
- On obtient finalement ~45% d'accuracy sur CIFAR-10 avec le mini-batching contre 18% sans. Ces résultats relativement faiblent s'expliquent par la simplicité du modèle : une seule couche cachée de 64 neurones.

