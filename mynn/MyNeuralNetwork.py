import math
import torch
from torch import tensor
import random
import os
import logging

logger = logging.getLogger(__name__)
class CantUseCuda(Exception):
    def __init__(self, message):
        super().__init__(message)
class UnknownFunction(Exception):
    pass
class NeuralNetworkMustHaveName(Exception):
    pass
class UCantFreezeAll(Exception):
    pass
class FileNotCorrect(Exception):
    def __init__(self, message):
        super().__init__(message)
class MyNeuralNetwork:
    w_init_functions = {}

    @classmethod
    def add_custom_w_init_function(cls, func_name):
        def decorator(func):
            def new_func(*args, **kwargs):
                val = func(*args, **kwargs)
                if not isinstance(val, (float, int)):
                    raise ValueError("Func must return a float")
                return val
            cls.w_init_functions[func_name] = new_func
            return new_func
        return decorator

    @staticmethod
    def simple_random(*args, **kwargs):
        return (-1) ** random.randint(0, 1) * random.random()
    w_init_functions["simple_random"] = simple_random
    @staticmethod
    def xavier(prev_level_size, cur_level_size):
        return random.gauss(0, math.sqrt(2 / (prev_level_size + cur_level_size)))
    w_init_functions["Xavier"] = xavier
    @staticmethod
    def Kaiming(prev_level_size, cur_level_size):
        return random.gauss(0, math.sqrt(2 / prev_level_size))
    w_init_functions["Kaiming"] = Kaiming
    def __init__(self,  levels_data: list, device: str = "cpu", neuralnetwork_name: str | None = None):
        if neuralnetwork_name is None:
            raise NeuralNetworkMustHaveName
        if device == "cpu":
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                 raise CantUseCuda("Ur device not support cuda")
        levels_sizes = [dat[0] for dat in levels_data]
        neurals_info = [None] * len(levels_sizes)
        neurals_w_init_functions_names = [None] * len(levels_sizes)
        for i in range(len(levels_data)):
            if len(levels_data[i]) > 1:
                neurals_info[i] = levels_data[i][1]
            if len(levels_data[i]) > 2:
                neurals_w_init_functions_names[i] = levels_data[i][2]


        self.levels_sizes = levels_sizes
        self.neurals_info = neurals_info
        self.neurals_w_init_functions_names = neurals_w_init_functions_names
        self.am_levels = len(self.levels_sizes)
        self.activate_coefs = []
        self.custom_funcs = {}
        self.error_type = "mse"
        self.am_neurals = 0
        for i in range(self.am_levels):
            ls = []
            for x in range(self.levels_sizes[i]):
                ls.append(self.w_init_functions["simple_random"]())
                self.am_neurals+=1
            self.activate_coefs.append(tensor(ls, requires_grad=True, device=self.device))

        self.all_weights = []

        for i in range(1, self.am_levels):
            neurals = []
            for j in range(self.levels_sizes[i]):
                ls = []
                for z in range(self.levels_sizes[i - 1]):
                    init_func_name = "simple_random"
                    if self.neurals_w_init_functions_names[i] is not None:
                        if self.neurals_w_init_functions_names[i] in self.w_init_functions:
                            init_func_name = self.neurals_w_init_functions_names[i]
                        else:
                            logger.error("We dont support this function,will be used simple_randon")
                    ls.append(self.w_init_functions[init_func_name](self.levels_sizes[i - 1], self.levels_sizes[i]))
                    self.am_neurals +=1
                neurals.append(ls)
            self.all_weights.append(tensor(neurals, requires_grad=True, device=self.device))
        neural_path = f"../networks/{neuralnetwork_name}"
        path_to_save =  os.path.join(os.path.dirname(__file__), neural_path)

        os.makedirs(path_to_save, exist_ok=True)
        self.neuralnetwork_dir = path_to_save
    def add_custom_func(self, func_name):
        def decorator(func):
            def hide(*args, **kwargs):
                val = func(*args, **kwargs)
                if type(val) == torch.Tensor:
                    return val
                raise ValueError("Custom func should return a tensor")
            self.custom_funcs[func_name] = hide
            return hide
        return decorator

    def save_weights(self, file_name: str):
        path = os.path.join(self.neuralnetwork_dir, file_name)
        data = ""
        for level_idx in range(0, self.am_levels):
            for neur_idx in range(self.levels_sizes[level_idx]):
                data += str(float(self.activate_coefs[level_idx][neur_idx]))
                data += ' '

        for level_idx in range(1, self.am_levels):
            for neur_idx in range(self.levels_sizes[level_idx]):
                for prev_neur_idx in range(self.levels_sizes[level_idx - 1]):
                    data += str(float(self.all_weights[level_idx - 1][neur_idx][prev_neur_idx]))
                    data += ' '
        data = data[:-1]

        with open(path, "w") as f:
            f.write(data)



    def load_weights(self, file_name):
        path = os.path.join(self.neuralnetwork_dir, file_name)
        try:
            with open(path, "r") as f:
                data = f.read().split(' ')

                if len(data) != self.am_neurals:
                    raise FileNotCorrect("File is not correct")
                data = list(map(float, data))
                pointer = 0
                for level_idx in range(0, self.am_levels):
                    for neur_idx in range(self.levels_sizes[level_idx]):
                        self.activate_coefs[level_idx].data[neur_idx] = data[pointer]
                        pointer += 1
                for level_idx in range(1, self.am_levels):
                    for neur_idx in range(self.levels_sizes[level_idx]):
                        for prev_neur_idx in range(self.levels_sizes[level_idx - 1]):
                            self.all_weights[level_idx - 1].data[neur_idx][prev_neur_idx] = data[pointer]
                            pointer+=1
        except FileNotFoundError:
            logger.error("File not found, cant load weights")
        except Exception as e:
            logger.error(f"Error {e}")
    def calc_error(self, output, correct_output):

        if self.error_type == "mse":
            return ((output - correct_output) ** 2).mean()
        if self.error_type == "cross_entropy":
            return torch.nn.functional.cross_entropy(output, correct_output)
        else:
            logger.error("We don't support this function, will be used mse")
            return ((output - correct_output) ** 2).mean()


    def calc_fun(self, z, level_idx):
        if self.neurals_info[level_idx] is None:
            return torch.sigmoid(z)
        func_name = self.neurals_info[level_idx]
        if func_name == "ReLU":
            return torch.relu(z)
        if func_name == "Linear":
            return z
        if func_name in self.custom_funcs:
            return self.custom_funcs[func_name](z)

        logger.error("We don't support this function, will be used sigmoid")
        return torch.sigmoid(z)
    def forward(self, input):
        neural_values = [input]
        for level_idx in range(1, self.am_levels):
            this_values = neural_values[level_idx - 1] @ self.all_weights[level_idx - 1].T
            this_values += self.activate_coefs[level_idx]
            this_values = self.calc_fun(this_values, level_idx)
            neural_values.append(this_values)
        return neural_values[-1]
    def one_step(self, input, correct_output):
        return self.calc_error(self.forward(input), correct_output)
    def train(self, epochs: int, learning_rate: float,  inputs: list | torch.Tensor, outputs: list | torch.Tensor, error_type: str = "mse",  bies_to_freeze: list | None = None, weights_to_freeze: list | None = None, optimizer_type: str = "SGD", batch_size: int | None = None):
        if bies_to_freeze is None:
            bies_to_freeze = []
        if weights_to_freeze is None:
            weights_to_freeze = []
        self.error_type = error_type
        all = []

        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        outputs = torch.as_tensor(outputs, dtype=torch.float32, device=self.device)
        N = len(inputs)
        bs = batch_size or N
        for level_idx in range(1, self.am_levels):
            if not (level_idx + 1) in weights_to_freeze:
                all.append(self.all_weights[level_idx - 1])
        for level_idx in range(self.am_levels):
            if not (level_idx + 1) in bies_to_freeze:
                all.append(self.activate_coefs[level_idx])
        if len(all) == 0:
            raise UCantFreezeAll
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params=all, lr=learning_rate)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(params=all, lr=learning_rate)
        else:
            logger.error("Unknown optimizer type, will be used SGD")
            optimizer = torch.optim.SGD(params=all, lr=learning_rate)

        for epoch in range(epochs):
            try:
                loss_counter = 0
                am_loss_calcs = 0
                perm = torch.randperm(N, device=self.device)

                for i in range(0, N, bs):
                    idx = perm[i:i + bs]
                    optimizer.zero_grad()
                    loss = self.one_step(inputs[idx], outputs[idx])
                    loss.backward()
                    optimizer.step()
                    loss_counter += loss.item()
                    am_loss_calcs+=1
                loss_counter/=am_loss_calcs
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: MSE: {loss_counter}")
            except KeyboardInterrupt:
                self.save_weights("autosave.txt")
                logger.info("Training stop")
                break
    def get_predict(self, input: list | torch.Tensor):
        input = torch.as_tensor(input,dtype=torch.float32, device=self.device)
        return self.forward(input)

