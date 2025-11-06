import numpy as np
import matplotlib.pyplot as plt
import read_cifar as rc
import warnings
warnings.filterwarnings("ignore")


def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    # --- Forward pass ---
    a0 = data
    z1 = np.matmul(a0, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))   # sigmoid hidden
    z2 = np.matmul(a1, w2) + b2
    a2 = 1 / (1 + np.exp(-z2))   # sigmoid output
    predictions = a2

    # Compute loss (mean squared error)
    loss = np.mean(np.square(predictions - targets))
    batch_size, d_out = predictions.shape

    # --- Backward pass ---
    # Output layer gradients
    d_a2 = 2 * (a2 - targets) / a2.size       # shape: (batch_size, d_out)
    d_z2 = d_a2 * a2 * (1 - a2)
    d_w2 = a1.T @ d_z2 / a2.size          # (d_h, d_out)
    d_b2 = np.mean(d_z2, axis=0, keepdims=True)

    # Hidden layer gradients
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * a1 * (1 - a1)
    d_w1 = a0.T @ d_z1 / batch_size          # (d_in, d_h)
    d_b1 = np.mean(d_z1, axis=0, keepdims=True)

    # --- Gradient descent step ---
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * d_b2
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * d_b1
    return w1, b1, w2, b2, loss


def one_hot(labels):
    num_classes = int(np.max(labels)) + 1
    one_hot_labels = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot_labels

def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    # Forward pass
    a0 = data
    z1 = np.matmul(a0, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid hidden layer
    z2 = np.matmul(a1, w2) + b2
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid output layer
    predictions = a2
    # One-hot encode labels for multi-class classification
    Y = one_hot(labels_train)
    # Binary cross-entropy loss
    eps = 1e-8  # Pour éviter log(0)
    loss = -np.mean(Y * np.log(predictions + eps) + (1 - Y) * np.log(1 - predictions + eps))
    # Backpropagation
    d_z2 = predictions - Y  # Admis dans l'énoncé
    d_w2 = a1.T @ d_z2 / data.shape[0]
    d_b2 = np.mean(d_z2, axis=0, keepdims=True)
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * a1 * (1 - a1)
    d_w1 = a0.T @ d_z1 / data.shape[0]
    d_b1 = np.mean(d_z1, axis=0, keepdims=True)
    # Mise à jour
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * d_b2
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * d_b1
    return w1, b1, w2, b2, loss

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = []
    for _ in range(num_epoch):
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        # Compute accuracy after update
        preds = np.argmax(1 / (1 + np.exp(-((1 / (1 + np.exp(-((data_train @ w1) + b1)))) @ w2 + b2))), axis=1)
        acc = np.mean(preds == labels_train)
        train_accuracies.append(acc)
    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    # Forward pass, puis argmax
    preds = np.argmax(1 / (1 + np.exp(-((1 / (1 + np.exp(-(data_test @ w1 + b1)))) @ w2 + b2))), axis=1)
    test_accuracy = np.mean(preds == labels_test)
    return test_accuracy

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    # Dimensions
    d_in = data_train.shape[1]
    d_out = int(np.max(labels_train)) + 1
    # Initialisation
    rng = np.random.default_rng(42)  # Pour reproductibilité
    w1 = rng.uniform(-1, 1, (d_in, d_h)).astype(np.float32)
    b1 = np.zeros((1, d_h), dtype=np.float32)
    w2 = rng.uniform(-1, 1, (d_h, d_out)).astype(np.float32)
    b2 = np.zeros((1, d_out), dtype=np.float32)
    # Entraînement
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    # Test
    final_test_acc = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    return train_accuracies, final_test_acc


if __name__ == "__main__":
    data, labels = rc.read_cifar('data/cifar-10-batches-py')
    data_train, labels_train, data_test, labels_test = rc.split_dataset(data, labels, 0.9)
    split_factor = 0.9
    d_h = 64
    learning_rate = 0.1
    num_epoch = 100
    train_accuracies, final_test_acc = run_mlp_training(
        data_train, labels_train, data_test, labels_test,
        d_h, learning_rate, num_epoch)
    print(f"Final test accuracy: {final_test_acc:.3f}")
    plt.figure()
    plt.plot(range(1, num_epoch+1), train_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("MLP Training Accuracy over Epochs")
    plt.grid(True)
    plt.savefig("Plots/mlp.png", bbox_inches='tight', dpi=150)

