import random, torch
from sklearn.datasets import load_digits
from mynn.MyNeuralNetwork import MyNeuralNetwork

random.seed(0); torch.manual_seed(0)


d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32)
Y = torch.nn.functional.one_hot(torch.tensor(d.target), 10).float()

perm = torch.randperm(len(X))
X, Y = X[perm], Y[perm]
X_train, Y_train = X[:1400], Y[:1400]
X_test, Y_test = X[1400:], Y[1400:]

net = MyNeuralNetwork(
    [(64,), (64, "ReLU", "Kaiming"), (10, "Linear", "Xavier")],
    neuralnetwork_name="digits",
)
net.train(200, 0.01, X_train, Y_train,
          error_type="cross_entropy", optimizer_type="Adam", batch_size=64)

pred = net.get_predict(X_test).argmax(dim=1)
acc = (pred == Y_test.argmax(dim=1)).float().mean().item()
print(f"Test accuracy: {acc:.2%}")