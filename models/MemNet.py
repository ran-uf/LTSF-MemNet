import torch


class MemoryBuffer(torch.nn.Module):
    def __init__(self, key_dim, value_dim, memory_size, sigma=1, kernel='Gaussian'):
        super(MemoryBuffer, self).__init__()
        m = torch.zeros((memory_size, memory_size))
        m[:-1, 1:] = torch.eye(memory_size - 1)
        self.II = torch.nn.Parameter(m, requires_grad=False)

        self.key = torch.nn.UninitializedBuffer()
        self.value = torch.nn.UninitializedBuffer()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_size = memory_size
        self.sigma = sigma

    def reset_memory(self, batch, device='cpu'):
        self.key = torch.zeros((batch, self.key_dim, self.memory_size), device=device)
        self.value = torch.zeros((batch, self.value_dim, self.memory_size), device=device)

    def read(self, x):
        # num_heads = 16
        # dis = self.key.reshape(x.shape[0], self.key_dim // num_heads, num_heads, self.memory_size) \
        #       - x.unsqueeze(-1).reshape(x.shape[0], self.key_dim // num_heads, num_heads, 1)
        # o = torch.exp(-self.sigma * torch.mean(dis ** 2, dim=1))
        # o = torch.einsum('abcd, abd->abc', self.value.reshape(x.shape[0], num_heads, self.value_dim // num_heads, self.memory_size), o)
        # return o.reshape(o.shape[0], -1)
        # o = torch.exp(-self.sigma * torch.mean((self.key - x.unsqueeze(-1)) ** 2, dim=1))
        o = torch.einsum('baf, ba->bf', self.key, x)
        o = torch.softmax(o, dim=1)
        o = torch.bmm(self.value, o.unsqueeze(-1))
        return o.squeeze(-1)

    def write(self, key, value):
        self.key = self.key @ self.II
        self.value = self.value @ self.II
        self.key[:, :, 0] = key
        self.value[:, :, 0] = value


class MemLinear(torch.nn.Module):
    def __init__(self, in_features, hidden_features, key_features, value_features, memory_size, sigma=1):
        super(MemLinear, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.key_features = key_features
        self.value_features = value_features
        # self.s_hidden = 0
        # self.e_hidden = self.hidden_features
        # self.s_query = self.hidden_features
        # self.e_query = self.hidden_features + self.key_features
        # self.s_key = self.hidden_features + self.key_features
        # self.e_key = self.hidden_features + self.key_features * 2
        # self.s_value = self.hidden_features + self.key_features * 2
        # self.e_value = self.hidden_features + self.key_features * 2 + self.value_features

        self.s_query = 0
        self.e_query = self.key_features
        self.s_key = self.key_features
        self.e_key = self.key_features * 2
        self.s_value = self.key_features * 2
        self.e_value = self.key_features * 2 + self.value_features
        self.buffer = MemoryBuffer(key_features, value_features, memory_size, sigma)
        # self.layer = torch.nn.Linear(in_features + hidden_features + value_features,
        #                              hidden_features + key_features + key_features + value_features)
        self.layer = torch.nn.Linear(in_features + 2 * key_features + 2 * value_features,
                                     2 * key_features + value_features)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, hidden_state, content):
        inp = torch.cat([x, self.dropout(torch.relu(hidden_state)), content], dim=1).contiguous()
        o = self.layer(inp)
        content = self.buffer.read(o[:, self.s_query: self.e_query])
        self.buffer.write(o[:, self.s_key: self.e_key], o[:, self.s_value: self.e_value])
        # return o[:, self.s_hidden: self.e_hidden], content
        return o, content

    def reset_memory(self, *args, **kwargs):
        self.buffer.reset_memory(*args, **kwargs)


# class MemLinear(torch.nn.Module):
#     def __init__(self, in_features, hidden_features, key_features, value_features, memory_size, sigma=1):
#         super(MemLinear, self).__init__()
#         self.in_features = in_features
#         self.hidden_features = hidden_features
#         self.key_features = key_features
#         self.value_features = value_features
#         self.s_hidden = 0
#         self.e_hidden = self.hidden_features
#         self.s_query = self.hidden_features
#         self.e_query = self.hidden_features + self.key_features
#         self.s_key = self.hidden_features + self.key_features
#         self.e_key = self.hidden_features + self.key_features * 2
#         self.s_value = self.hidden_features + self.key_features * 2
#         self.e_value = self.hidden_features + self.key_features * 2 + self.value_features
#         self.buffer = MemoryBuffer(key_features, value_features, memory_size, sigma)
#         self.layer = torch.nn.Linear(in_features + hidden_features + value_features,
#                                      hidden_features + key_features + key_features + value_features)
#
#     def forward(self, x, hidden_state, content):
#         inp = torch.cat([x, hidden_state, content], dim=1).contiguous()
#         o = self.layer(inp)
#         content = self.buffer.read(o[:, self.s_query: self.e_query])
#         self.buffer.write(o[:, self.s_key: self.e_key], o[:, self.s_value: self.e_value])
#         return o[:, self.s_hidden: self.e_hidden], content
#
#     def reset_memory(self, *args, **kwargs):
#         self.buffer.reset_memory(*args, **kwargs)


class MemGRU(torch.nn.Module):
    def __init__(self, in_features, hidden_features, key_features, value_features, memory_size, sigma=1):
        super(MemGRU, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.key_features = key_features
        self.value_features = value_features
        self.s_hidden = 0
        self.s_query = 0
        self.e_query = self.key_features
        self.s_key = self.key_features
        self.e_key = self.key_features * 2
        self.s_value = self.key_features * 2
        self.e_value = self.key_features * 2 + self.value_features
        self.buffer = MemoryBuffer(key_features, value_features, memory_size, sigma)
        self.layer = torch.nn.GRUCell(in_features + value_features,
                                      key_features + key_features + value_features)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, hidden_state, content):
        inp = torch.cat([x, content], dim=1).contiguous()
        o = self.layer(inp, self.dropout(hidden_state))
        content = self.buffer.read(o[:, self.s_query: self.e_query])
        self.buffer.write(o[:, self.s_key: self.e_key], o[:, self.s_value: self.e_value])
        return o, content

    def reset_memory(self, *args, **kwargs):
        self.buffer.reset_memory(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_features = args.enc_in
        self.hidden_features = 2 * args.key_features + args.value_features
        # self.hidden_features = args.value_features
        self.value_features = args.value_features
        self.layer = MemLinear(args.enc_in, self.hidden_features, args.key_features, args.value_features, args.memory_size, args.sigma)
        self.s_output = torch.nn.Linear(self.hidden_features, args.c_out)
        self.c_output = torch.nn.Linear(args.value_features, args.c_out)
        self.init_hidden = torch.nn.Parameter(torch.randn(1, self.hidden_features) * 0.1, requires_grad=True)
        self.init_content = torch.nn.Parameter(torch.randn(1, self.value_features) * 0.1, requires_grad=True)

    def forward(self, x, predict_length=1):
        o = []
        length = x.shape[1]
        self.layer.reset_memory(x.shape[0], x.device)
        state = self.init_hidden.repeat(x.shape[0], 1)
        content = self.init_content.repeat(x.shape[0], 1)
        for i in range(length):
            state, content = self.layer(x[:, i], state, content)
            if self.training:
                o.append(self.s_output(state) + self.c_output(content))

        pred = self.s_output(state) + self.c_output(content)
        if not self.training:
            o.append(pred)
        for i in range(predict_length - 1):
            state, content = self.layer(pred, state, content)
            pred = self.s_output(state) + self.c_output(content)
            o.append(pred)
        if self.training:
            return torch.stack(o, dim=1)
        else:
            return torch.stack(o, dim=1)
