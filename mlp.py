import numpy as np 
import matplotlib.pyplot as plt 
from read_cifar import read_cifar, split_dataset
from warnings import filterwarnings
filterwarnings("ignore")

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Fonction sigmoïde appliquée élément par élément."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Dérivée de la sigmoïde en fonction de sa sortie a."""
    return a * (1 - a)
    

def one_hot(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    Convertit un vecteur de labels en matrice one-hot.
    
    Paramètres :
    labels (np.ndarray) : vecteur de labels de taille (batch_size,)
    num_classes (int) : nombre de classes (optionnel, calculé à partir des labels si None)
    
    Retour :
    one_hot_matrix (np.ndarray) : matrice de taille (batch_size, num_classes)
    """
    labels = np.array(labels, dtype=int)
    if num_classes is None:
        num_classes = labels.max() + 1
    one_hot_matrix = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return one_hot_matrix

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    """
    Effectue une étape de gradient descent pour un MLP à 1 hidden layer
    avec MSE comme fonction de coût (pour la régression).
    
    Paramètres :
    w1, b1 : poids et biais de la couche cachée
    w2, b2 : poids et biais de la couche de sortie
    data : matrice d'entrée (batch_size x d_in)
    targets : matrice de sorties désirées (batch_size x d_out)
    learning_rate : taux d'apprentissage
    
    Retour :
    w1, b1, w2, b2 : poids et biais mis à jour
    loss : valeur du MSE pour monitoring
    """
    # Forward pass
    a0 = data
    z1 = a0 @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Calcul du loss (MSE)
    loss = np.mean((a2 - targets) ** 2)
    
    # Backpropagation
    batch_size = data.shape[0]
    d_out = targets.shape[1]


    dC_da2 = 2 * (a2 - targets) / (batch_size * d_out)  # gradient MSE
    da2_dz2 = sigmoid_derivative(a2)
    dz2_dw2 = a1.T
    
    dz2 = dC_da2 * da2_dz2  # ∂C/∂Z2
    dw2 = dz2.T @ a1
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    dz1 = (dz2 @ w2.T) * sigmoid_derivative(a1)  # ∂C/∂Z1
    dw1 = a0.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Update weights
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2.T
    b2 -= learning_rate * db2

    return w1, b1, w2, b2, loss

def learn_once_cross_entropy(w1, b1, w2, b2, data, targets, learning_rate):
    """
    Effectue une étape de gradient descent pour un MLP à 1 hidden layer
    avec cross-entropy comme fonction de coût (classification multi-classes).
    Utilise softmax pour la couche de sortie.
    
    Paramètres :
    w1, b1 : poids et biais de la couche cachée
    w2, b2 : poids et biais de la couche de sortie
    data : matrice d'entrée (batch_size x d_in)
    tragets : vecteur de labels (batch_size, d_out))
    learning_rate : taux d'apprentissage
    
    Retour :
    w1, b1, w2, b2 : poids et biais mis à jour
    loss : valeur du cross-entropy pour monitoring
    """
    batch_size = data.shape[0]
    num_classes = np.max(targets) + 1
    Y = targets 

    # Forward pass
    a0 = data
    z1 = a0 @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    
    # Softmax
    exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    d_out = Y.shape[1]
    # print(a2.shape, Y.shape)

    # Loss (cross-entropy)
    loss = -np.mean(np.sum(Y * np.log(a2 + 1e-8), axis=1))
    
    # Backpropagation
    dz2 = (a2 - Y) / (batch_size)  
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ w2.T) * sigmoid_derivative(a1)
    dw1 = a0.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Update weights
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    
    return w1, b1, w2, b2, loss

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    """
    Entraîne le MLP pendant num_epoch étapes.
    Retourne poids/biais mis à jour et accuracies d'entraînement par epoch.
    """
    train_accuracies = []
    batch_size = 64
    for epoch in range(1, num_epoch + 1):
        for i in range(0, len(data_train), batch_size):
            batch_data = data_train[i: i + batch_size]
            batch_labels = labels_train[i: i + batch_size]
            w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, batch_data, batch_labels, learning_rate)
        # Prédiction sur train pour accuracy
        z1 = data_train @ w1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ w2 + b2
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        preds = np.mean(np.argmax(a2, axis=1) == np.argmax(labels_train, axis=1))
        train_accuracies.append(preds)
        if epoch % 10 == 0 :
            print(f"Epoch {epoch}: loss={loss:.4f}, train_accuracy={preds:.4f}")
    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    """
    Teste le MLP et retourne l'accuracy sur les données test.
    """
    # Layer 1
    z1 = data_test @ w1 + b1
    a1 = sigmoid(z1)
    # Layer 2
    z2 = a1 @ w2 + b2
    exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    preds = np.argmax(a2, axis=1)
    # print(preds.shape, labels_test.shape)
    accuracy = np.mean(preds == labels_test)

    return accuracy

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h=64, learning_rate=0.1, num_epoch=100):
    """
    Entraîne un MLP et retourne l'évolution des accuracies et l'accuracy finale test.
    """
    d_in = data_train.shape[1]
    num_classes = np.max(labels_train) + 1
    d_out = num_classes
    labels_train = one_hot(labels_train, num_classes)
    
    # Initialisation aléatoire
    w1 = 2 * np.random.rand(d_in, d_h) - 1
    b1 = np.zeros((1, d_h))
    w2 = 2 * np.random.rand(d_h, d_out) - 1
    b2 = np.zeros((1, d_out))

    data_train = data_train / 255.0
    data_test = data_test / 255.0
    
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)

    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    
    return train_accuracies, test_accuracy


if __name__ == "__main__":

    # Test forward pass

    N = 30  # number of input data
    d_in = 3  # input dimension
    d_h = 3  # number of neurons in the hidden layer
    d_out = 2  # output dimension (number of neurons of the output layer)

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    data = np.random.rand(N, d_in)  # create a random data
    targets = np.random.rand(N, d_out)  # create a random targets

    # Forward pass
    a0 = data # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    predictions.shape

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))
    print("\ntest de forward pass :\n")
    print(loss)

    # Variables random pour les tests

    # Paramètres du mini-Réseau
    N = 5         # nombre d'exemples
    d_in = 3      # taille entrée
    d_h = 4       # taille couche cachée
    d_out = 3     # nombre de classes

    # Initialisation aléatoire des poids et biais
    w1 = np.random.rand(d_in, d_h)
    b1 = np.random.rand(1, d_h)
    w2 = np.random.rand(d_h, d_out)
    b2 = np.random.rand(1, d_out)

    # Données et labels factices
    data_test = np.random.rand(N, d_in)
    labels_test = np.random.randint(0, d_out, size=(N,))

    data_train = np.random.rand(N, d_in)
    labels_train = np.random.randint(0, d_out, size=(N,))
    labels_train_one_hot = one_hot(labels_train, num_classes=d_out)

    learning_rate = 0.1
    num_epoch = 9

    # Test de la fonction learn_once_mse
    w1, b1, w2, b2, loss = learn_once_mse(w1, b1, w2, b2, data_train, labels_train_one_hot, learning_rate)
    print("\nTest learn_once_mse :\n")
    print("Loss:", loss)
    print("w1 shape:", w1.shape)
    print("b1 shape:", b1.shape)
    print("w2 shape:", w2.shape)
    print("b2 shape:", b2.shape)

    # Test de la fonction learn_once_cross_entropy
    w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train_one_hot, learning_rate)
    print("\nTest learn_once_cross_entropy :\n")
    print("Loss:", loss)
    print("w1 shape:", w1.shape)
    print("b1 shape:", b1.shape)
    print("w2 shape:", w2.shape)
    print("b2 shape:", b2.shape)


    # Test de la fonction train_mlp
    w1, b1, w2, b2, train_accuracies = train_mlp(
        w1, b1, w2, b2, data_train, labels_train_one_hot, learning_rate, num_epoch
)
    print("\ntest de train_mlp :\n")
    print(train_accuracies)


    # Test de la fonction test_mlp
    accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    print("\ntest de test_mlp :\n")
    print(accuracy)


    # Test one_hot function
    print("\ntest de one_hot :\n")
    print(one_hot(np.array([1, 2, 0])))

    # Charger CIFAR
    all_data, all_labels = read_cifar("data/cifar-10-batches-py")

    data_train, labels_train, data_test, labels_test = split_dataset(all_data, all_labels, split=0.9)

    # Paramètres
    d_h = 64
    learning_rate = 0.1
    num_epoch = 100
    
    print("\nEntraînement du MLP sur CIFAR-10 :\n")
    # Entraînement
    train_accuracies, test_accuracy = run_mlp_training(
        data_train, labels_train, data_test, labels_test,
        d_h=d_h, learning_rate=learning_rate, num_epoch=num_epoch
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    # Plot learning curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epoch+1), train_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("MLP Training Accuracy")
    plt.grid(True)
    plt.savefig("results/mlp.png")

