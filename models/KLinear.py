import torch


# class _ParameterMeta(torch._C._TensorMeta):
#     # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
#     def __instancecheck__(self, instance):
#         return super().__instancecheck__(instance) or (
#             isinstance(instance, torch.Tensor) and getattr(instance, '_is_param', False))
#
#
# class KernelTensor(torch.Tensor):
#     @staticmethod
#     def __new__(cls, x, in_dim, out_dim, *args, **kwargs):
#         return super().__new__(cls, x, *args, **kwargs)
#
#     def __init__(self, x, dim_in, dim_out):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.entries_forward = []
#         self.entries_backward = []


class KernelParameter(torch.nn.Parameter):
    pass


class KernelLayerBase(torch.nn.Module):
    pass


class KernelModule(torch.nn.Module):
    def kernel_parameters(self):
        parameters = torch.nn.ModuleList()
        for m in self.modules():
            if isinstance(m, KernelLayerBase):
                parameters.append(m)
        return parameters

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if not isinstance(param, KernelParameter):
                yield param


def gaussian(x: torch.Tensor, y: torch.Tensor, sigma=1):
    return torch.exp(-sigma * torch.mean((x - y) ** 2, dim=-1))


def batch_gaussian(x, y, sigma=1):
    return gaussian(x.transpose(0, -2), y, sigma).transpose(0, -1)


class KernelLinear(KernelLayerBase):
    def __init__(self, in_features, out_features, sigma=1):
        super(KernelLinear, self).__init__()
        if type(in_features) == int:
            self.in_features = [(0, in_features)]
            self.sigma = [sigma]
        elif type(in_features) == list:
            self.in_features = []
            start = 0
            for i in range(len(in_features)):
                self.in_features.append((start, start + in_features[i]))
                start += in_features[i]
            self.sigma = sigma
        self.out_features = out_features
        self.total_in_features = in_features if type(in_features) == int else sum(in_features)
        key = -1000 * torch.ones(1, self.total_in_features)
        value = torch.zeros(1, out_features)
        self.register_parameter('w', KernelParameter(torch.cat([key, value], dim=1)))
        self.forward_entries = []
        self.backward_entries = []
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_full_backward_hook(self._backward_hook)

    def extra_repr(self) -> str:
        return 'num_kernels={}, in_features={}, out_features={}'.format(
            self.w.shape[0], self.in_features, self.out_features
        )

    def _forward_pre_hook(self, module, inp):
        # print(module, inp)
        self.forward_entries.append(inp[0].detach().data)
        if len(self.forward_entries) > 30:
            self.forward_entries.remove(self.forward_entries[0])

    def _backward_hook(self, module, p1, p2):
        if len(self.backward_entries) < 30:
            self.backward_entries.append(p2[0].detach().data)

    @staticmethod
    def batch_gaussian(w, x, sigma):
        return gaussian(w.unsqueeze(0), x.unsqueeze(1), sigma)

    def forward(self, x):
        o = self.batch_gaussian(self.w[:, self.in_features[0][0]:self.in_features[0][1]],
                                x[:, self.in_features[0][0]:self.in_features[0][1]],
                                sigma=self.sigma[0])
        for i in range(1, len(self.in_features)):
            o = o * self.batch_gaussian(self.w[:, self.in_features[i][0]:self.in_features[i][1]],
                                        x[:, self.in_features[i][0]:self.in_features[i][1]],
                                        sigma=self.sigma[i])
        return o @ self.w[:, -self.out_features:]


class KernelLayer(KernelModule):
    def __init__(self, in_features, out_features, sigma=1):
        super(KernelLayer, self).__init__()
        self.kernel_linear = KernelLinear(in_features, out_features, sigma)

    def forward(self, x):
        return self.kernel_linear(x[:, :, 0]).unsqueeze(2)


class KernelConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', device=None, dtype=None, sigma=1):
        super(KernelConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.sigma = sigma

        self.unfold = torch.nn.Unfold(self.kernel_size, dilation, padding, stride)
        self.kernel_linear = KernelLinear(kernel_size[0] * kernel_size[1] * self.in_channels, self.out_channels, self.sigma)

    def forward(self, x):
        batch, _, w, h = x.shape
        x = self.unfold(x)
        features = x.shape[1]
        x = self.kernel_linear(x.transpose(1, 2).reshape(-1, features)).reshape(batch, -1, self.out_channels).transpose(1, 2)
        return x.view(batch, x.shape[1],
                      int((w + 2 * self.padding - self.dilation * (self.kernel_size[0] - 1) - 1) / self.stride + 1),
                      int((w + 2 * self.padding - self.dilation * (self.kernel_size[1] - 1) - 1) / self.stride + 1))


class KernelRNNCell(KernelModule):
    def __init__(self, input_dim, hidden_dim, sigma):
        super(KernelRNNCell, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.kernel_linear = KernelLinear([input_dim, hidden_dim], hidden_dim, sigma)

    def forward(self, x, state):
        if state is None:
            state = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        o = self.kernel_linear(torch.cat([x, state], dim=1))
        return o, o


class KernelRNN(KernelModule):
    def __init__(self, input_dim, hidden_dim, output_dim, sigma):
        super(KernelRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = KernelRNNCell(input_dim, hidden_dim, sigma)
        self.o = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.o.weight.data = torch.zeros(self.o.weight.data.shape)
        self.o.weight.data[-output_dim:, -output_dim:] = torch.eye(output_dim)
        self.ini_hidden = torch.randn(1, hidden_dim)

    def extra_repr(self) -> str:
        return 'KNN num_kernels={}'.format(
            self.cell.kernel_linear.w.shape[0]
        )

    def forward(self, x, state=None):
        o = []
        if not state:
            state = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)
            # state = self.ini_hidden.repeat(x.shape[0], 1).to(x.device)
        for i in range(x.shape[1]):
            _o, state = self.cell(x[:, i], state)
            o.append(_o)
        return self.o(torch.stack(o, dim=1)), state


class KernelOptimizer(object):
    def __init__(self, parameters: torch.nn.ParameterList, lr=0.01, quantization=[7e-1], *args, **kwargs):
        # super(KernelOptimizer, self).__init__(layers, kwargs)
        self.parameters = parameters
        self.quantization = quantization
        self.lr = lr

    def step(self, *arg, **kwargs):
        for parameter in self.parameters:
            self._step(parameter)

            def _clear_redundant(par):
                index = ((par.w.data[:, -par.out_features:] ** 2).sum(dim=1) > 1e-8).nonzero()[:, 0]
                if index.size()[0] > 0:
                    par.w.data = par.w.data[index]

            # if parameter.w.data.shape[0] > 100:
            #     _clear_redundant(parameter)

    def _step(self, parameter):
        forward_entries = torch.stack(parameter.forward_entries, dim=1).reshape(-1, parameter.total_in_features)
        backward_entries = torch.stack(parameter.backward_entries[::-1], dim=1).reshape(-1, parameter.out_features)
        grad = torch.cat([forward_entries, -self.lr * backward_entries], dim=1)

        def _sim(x, y, features, q, sigma, diagonal=None):
            with torch.no_grad():
                re = torch.mean((x[:, features[0][0]:features[0][1]].unsqueeze(1) - y[:, features[0][0]:features[0][1]].unsqueeze(0)) ** 2, dim=-1)
                if diagonal is not None:
                    re += (q[0] / sigma[0] + 1) * torch.triu(torch.ones(re.shape, device=x.device), diagonal=diagonal)
                o = re < q[0] / sigma[0]
                for i in range(1, len(sigma)):
                    re = torch.sum((x[:, features[i][0]:features[i][1]].unsqueeze(1) - y[:, features[i][0]:features[i][1]].unsqueeze(0)) ** 2,
                                   dim=-1)
                    if diagonal is not None:
                        re += (q[i] / sigma[i] + 1) * torch.triu(torch.ones(re.shape, device=x.device), diagonal=diagonal)
                    o *= re < q[i] / sigma[i]
            return o

        def self_merge(x, split, in_features, sigma):
            # sim = torch.mean((x[:, :split].unsqueeze(1) - x[:, :split].unsqueeze(0)) ** 2, dim=-1)
            # sim = sim + torch.triu(torch.ones(sim.shape, device=x.device), diagonal=0) < self.quantization
            sim = _sim(x[:, :split], x[:, :split], in_features, self.quantization, sigma, 0)
            indices = sim.nonzero()
            # print(indices.shape[0])
            if indices.shape[0] == 0:
                return x
            else:
                indices = indices.cpu().numpy().tolist()
                # print(indices)
                mp_i = []
                mp_j = []
                while indices:
                    [temp_i, temp_j] = indices.pop()
                    mp_i.append(temp_i)
                    mp_j.append(temp_j)
                    indices = [i for i in indices if i[1] != temp_i and i[0] != temp_i
                                                 and i[1] != temp_j and i[0] != temp_j]
                    # for _i in indices:
                    #     if _i[1] == temp_i or _i[0] == temp_i:
                    #         indices.remove(_i)
                for i, j in zip(mp_i, mp_j):
                    x[j, split:] += x[i, split:]
                ids = [i for i in range(x.shape[0])]
                # print(ids, mp_i)
                for i in mp_i:
                    ids.remove(i)
                return x[ids]

        # merge x to y
        def merge(x, y, split, in_features, sigma):
            # sim = torch.mean((x[:, :split].unsqueeze(1) - y[:, :split].unsqueeze(0)) ** 2, dim=-1)
            # sim = sim < self.quantization
            sim = _sim(x[:, :split], y[:, :split], in_features, self.quantization, sigma)
            indices = sim.nonzero()
            if indices.shape[0] == 0:
                return torch.cat([x, y], dim=0)
            else:
                indices = indices.cpu().numpy().tolist()
                mp_i = []
                mp_j = []
                while indices:
                    [temp_i, temp_j] = indices.pop()
                    mp_i.append(temp_i)
                    mp_j.append(temp_j)
                    indices = [i for i in indices if i[0] != temp_i]

                for i, j in zip(mp_i, mp_j):
                    y[j, split:] += x[i, split:]
                ids = [i for i in range(x.shape[0])]
                for i in mp_i:
                    ids.remove(i)
                return torch.cat([y, x[ids]], dim=0)

        grad = self_merge(grad, parameter.total_in_features, parameter.in_features, parameter.sigma)
        # print('grad', grad.shape)
        parameter.w.data = merge(grad, parameter.w.data, parameter.total_in_features, parameter.in_features, parameter.sigma)
        # print('merge', parameter.w.data.shape)
        parameter.w.data = self_merge(parameter.w.data, parameter.total_in_features, parameter.in_features, parameter.sigma)
        # print('self merge', parameter.w.data.shape)

    def zero_grad(self, *args, **kwargs):
        for parameter in self.parameters:
            self._zero_grad(parameter)

    @staticmethod
    def _zero_grad(parameter):
        parameter.forward_entries = []
        parameter.backward_entries = []
