import random
import torch

from mynn.MyNeuralNetwork import MyNeuralNetwork
random.seed(42)
def prepare_data(cnt_cases: int):
    X = []
    Y = []
    for case_idx in range(cnt_cases):
        mas = []
        for i in range(5):
            mas.append(random.randint(-100, 100))
        X.append(mas)
        Y.append([sum(mas)])
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    return X, Y

net = MyNeuralNetwork(
    [(5,), (1, "Linear", "Xavier")], device="cpu", neuralnetwork_name="summator"
)

X, Y = prepare_data(cnt_cases=500)
net.train(200, 0.02, X, Y, error_type="mse", optimizer_type="Adam", batch_size=20)

X_test, Y_test = prepare_data(cnt_cases=500)


pred = net.get_predict(X_test)
mae = (pred - Y_test).abs().mean().item()
rmse = ((pred - Y_test) ** 2).mean().sqrt().item()
print(f"MAE: {mae:.3e}, RMSE: {rmse:.3e}")