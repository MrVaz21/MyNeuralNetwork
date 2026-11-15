import random, torch
from mynn.MyNeuralNetwork import MyNeuralNetwork

def make_net(sizes, name="test"):
    return MyNeuralNetwork(sizes, neuralnetwork_name=name)

def test_weight_shapes():
    net = make_net([(3,), (5, "ReLU"), (2, "Linear")])
    assert net.all_weights[0].shape == (5, 3)
    assert net.all_weights[1].shape == (2, 5)

def test_forward_batch_shape():
    net = make_net([(3,), (4, "ReLU"), (2, "Linear")])
    out = net.get_predict(torch.rand(7, 3))
    assert out.shape == (7, 2)

def test_save_load_roundtrip():
    net = make_net([(2,), (3, "ReLU"), (1, "Linear")], "roundtrip")
    x = torch.rand(5, 2)
    before = net.get_predict(x)
    net.save_weights("w.txt")
    net2 = make_net([(2,), (3, "ReLU"), (1, "Linear")], "roundtrip")
    net2.load_weights("w.txt")
    assert torch.allclose(before, net2.get_predict(x), atol=1e-5)

def test_learns_addition():
    random.seed(0); torch.manual_seed(0)
    net = make_net([(3,), (1, "Linear")], "add")
    X = torch.rand(300, 3) * 2 - 1
    Y = X.sum(dim=1, keepdim=True)
    net.train(1500, 0.01, X, Y, optimizer_type="Adam")
    mae = (net.get_predict(X) - Y).abs().mean().item()
    assert mae < 1e-2