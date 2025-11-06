import numpy as np
import pickle


def read_cifar_batch(batch_path="data/cifar-10-batches-py/data_batch_1"):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32)
    labels = np.array(batch[b'labels'], dtype=np.int64)
    return data, labels

def read_cifar(directory_path="data/cifar-10-batches-py"):
    data_list = []
    labels_list = []
    for i in range(1, 6):
        batch_path = f"{directory_path}/data_batch_{i}"
        data, labels = read_cifar_batch(batch_path)
        data_list.append(data)
        labels_list.append(labels)
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_data, all_labels

def split_dataset(data : np.ndarray, labels : np.ndarray , split : float):
    num_samples = data.shape[0]
    split_index = int(num_samples * split)
    data_train = data[:split_index]
    labels_train = labels[:split_index]
    data_test = data[split_index:]
    labels_test = labels[split_index:]
    return data_train, labels_train, data_test, labels_test

if __name__ == "__main__":
    batch_path = "data/cifar-10-batches-py/data_batch_1"
    data, labels = read_cifar_batch(batch_path)
    print(data.shape)
    print(data.dtype)

    directory_path = "data/cifar-10-batches-py"
    all_data, all_labels = read_cifar(directory_path)
    print(all_data.shape)

    split = split_dataset(all_data, all_labels, split=0.8)
    print(split[0].shape)
