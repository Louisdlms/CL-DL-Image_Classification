# CL-DL-Image_Classification

**Projet**: Classification d'images (CIFAR-10) — implémentations pédagogiques de KNN et d'un MLP

Ce dépôt contient des scripts pédagogiques pour expérimenter deux approches de classification d'images sur le jeu CIFAR-10 :
- un classifieur K-Nearest Neighbors (`knn.py`) ;
- un perceptron multi-couches simple (MLP) entraîné par gradient descendant (`mlp.py`).

**But**: fournir un code clair et éducatif pour comprendre l'entraînement et l'évaluation de modèles basiques sur des données visuelles.

**Données**
- **Emplacement**: les fichiers CIFAR-10 doivent être présents dans `data/cifar-10-batches-py` (les fichiers `data_batch_1..5` et `test_batch`).
- **Téléchargement**: si vous n'avez pas les fichiers, téléchargez CIFAR-10 depuis https://www.cs.toronto.edu/~kriz/cifar.html puis placez l'archive extraite dans `data/`.

**Installation**
- **Requis**: Python 3.8+ recommandé.
- **Dépendances**: installer depuis le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

- Remarque: si l'installation échoue à cause d'un package `matplot` introuvable, installez `matplotlib` manuellement :

```bash
pip install numpy matplotlib
```

**Usage rapide**
- Pour tester la lecture des fichiers CIFAR:

```bash
python3 read_cifar.py
```

- Pour exécuter l'expérience KNN (génère `results/knn.png`):

```bash
python3 knn.py
```

- Pour entraîner et tester le MLP sur CIFAR-10 (génère `results/mlp.png`):

```bash
python3 mlp.py
```

- Les hyperparamètres (par ex. `d_h`, `learning_rate`, `num_epoch`) sont définis dans le bloc `if __name__ == "__main__"` de `mlp.py` et peuvent être modifiés directement.

**Structure du dépôt**
- `mlp.py` : implémentation d'un MLP à une couche cachée, fonctions d'entraînement, test et script d'entraînement complet sur CIFAR-10.
- `knn.py` : implémentation vectorisée de KNN (distance L2, vote majoritaire) et script d'évaluation sur CIFAR-10.
- `read_cifar.py` : utilitaires pour lire les fichiers CIFAR-10, concaténer et séparer train/test.
- `requirements.txt` : dépendances Python (voir remarque ci‑dessus).
- `data/` : dossier attendu contenant `cifar-10-batches-py`.
- `results/` : dossiers de sortie pour graphiques (`knn.png`, `mlp.png`).

**Conseils et notes**
- Les scripts sont conçus pour être pédagogiques plutôt qu'optimisés pour la performance (par ex. pas d'utilisation de GPU).
- Le MLP utilise un mini-batching simple : `batch_size = 64` (voir `train_mlp` dans `mlp.py`) et les gradients sont calculés sur chaque mini-batch. Les données ne sont cependant pas reshufflées automatiquement à chaque epoch (le mélange initial est effectué par `split_dataset` dans `read_cifar.py`).
- Pour des entraînements plus robustes et rapides, envisagez d'utiliser `PyTorch`/`TensorFlow` et des optimisations supplémentaires (shuffle par epoch, mini-batches plus petits/grands selon l'optimiseur, normalisation, GPU).
- Le fichier `requirements.txt` contient `matplot` (probable coquille). Si vous rencontrez une erreur, remplacez `matplot` par `matplotlib`.

**Analyse des résultats**

- **KNN (`knn.py`)** :
	- Sortie attendue : un graphique `results/knn.png` affichant l'accuracy en fonction de `k` (nombre de voisins).
	- Interprétation : lorsque vous utilisez les pixels bruts de CIFAR-10 comme caractéristiques, le KNN arrive généralement à des performances modestes (les images couleur non transformées sont difficiles à séparer par distance euclidienne simple). L'accuracy variera avec `k` — de petites valeurs de `k` peuvent être sensibles au bruit, des valeurs trop grandes peuvent lisser trop les classes.
	- Pistes d'amélioration : extraction de caractéristiques (PCA, HOG), utilisation de descripteurs appris (features CNN), normalisation ou réduction de dimension, ou bien des variantes de distance/poids.

- **MLP (`mlp.py`)** :
	- Sortie attendue : un graphique `results/mlp.png` montrant l'évolution de l'accuracy d'entraînement par epoch et un affichage de l'accuracy sur le jeu de test.
	- Observations typiques :
		- la précision d'entraînement augmente au fil des epochs ; l'accuracy de test est généralement inférieure (écart dû à l'overfitting si le modèle est trop puissant ou mal régularisé).
		- la vitesse d'apprentissage et la qualité finale dépendent fortement de l'architecture (`d_h`), du taux d'apprentissage (`learning_rate`), du `batch_size` et du nombre d'epochs (`num_epoch`).
	- Pistes d'amélioration : shuffle des données à chaque epoch, normalisation des entrées (déjà divisées par 255 dans le script), régularisation (weight decay, dropout), optimiser l'initialisation, ou remplacer le MLP par un CNN pour des performances nettement supérieures sur CIFAR-10.


**Exemples de commandes utiles**
- Installer dépendances : `pip install -r requirements.txt` ou `pip install numpy matplotlib`.
- Lancer le MLP : `python3 mlp.py`.
- Lancer KNN : `python3 knn.py`.

**Contact / Auteurs**
- TP de Deep Learning — Centrale Lyon.

