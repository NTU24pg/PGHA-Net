import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from temporal_self_attention import TemporalSelfAttentionLayer


class DAGCL(nn.Module):
    def __init__(self, channels: int, pdim_ratio: float = 0.5, diffusion_order: int = 2, dropout: float = 0.1):
        super().__init__()
        assert diffusion_order >= 1
        self.channels = channels
        self.diffusion_order = diffusion_order
        self.pdim = max(1, int(channels * pdim_ratio))
        self.dynamic_gate = nn.Conv1d(in_channels=channels, out_channels=self.pdim, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def _graph_aggregate(self, x, adj):
        if adj.dim() == 3:
            return torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
        return torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()

    def forward(self, x, adj):
        x_k = self._graph_aggregate(x, adj)
        x_static = x_k.clone()

        for _ in range(1, self.diffusion_order):
            x_k = self._graph_aggregate(x_k, adj)
            x_static = x_static + x_k

        x_pool = x.mean(-1)
        gate = torch.sigmoid(self.dynamic_gate(x_pool))
        x_one_hop = self._graph_aggregate(x, adj)
        x_dynamic = gate.unsqueeze(-1) * x_one_hop[:, :self.pdim, :, :]

        x_fused = torch.cat(
            [
                x_static[:, :self.pdim, :, :] + x_dynamic,
                x_static[:, self.pdim:, :, :]
            ],
            dim=1
        )

        out = self.output_projection(x_fused)
        return self.dropout(out)


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.conv3(self.dropout(self.conv1(x) * torch.sigmoid(self.conv2(x))))


class ITPE(nn.Module):
    def __init__(self, time_slots, features):
        super().__init__()
        self.time_slots = time_slots

        self.time_of_day_embedding = nn.Parameter(torch.empty(time_slots, features))
        self.day_of_week_embedding = nn.Parameter(torch.empty(7, features))

        self.periodic_day_embedding_1 = nn.Parameter(torch.empty(time_slots, features))
        self.periodic_day_embedding_2 = nn.Parameter(torch.empty(time_slots, features))
        self.periodic_week_embedding_1 = nn.Parameter(torch.empty(7, features))
        self.periodic_week_embedding_2 = nn.Parameter(torch.empty(7, features))

        nn.init.xavier_uniform_(self.time_of_day_embedding)
        nn.init.xavier_uniform_(self.day_of_week_embedding)

        for parameter in [
            self.periodic_day_embedding_1,
            self.periodic_day_embedding_2,
            self.periodic_week_embedding_1,
            self.periodic_week_embedding_2
        ]:
            nn.init.xavier_uniform_(parameter)

    def forward(self, x, return_periodic_guidance=False):
        day_index = (x[..., 1] * self.time_slots).floor().clamp(0, self.time_slots - 1).long().to(x.device)
        week_index = x[..., 2].clamp(0, 6).long().to(x.device)

        time_of_day = self.time_of_day_embedding[day_index]
        day_of_week = self.day_of_week_embedding[week_index]
        combined_temporal_embedding = (time_of_day + day_of_week).permute(0, 3, 2, 1).contiguous()

        if not return_periodic_guidance:
            return combined_temporal_embedding

        periodic_day_1 = self.periodic_day_embedding_1[day_index]
        periodic_day_2 = self.periodic_day_embedding_2[day_index]
        periodic_week_1 = self.periodic_week_embedding_1[week_index]
        periodic_week_2 = self.periodic_week_embedding_2[week_index]

        periodic_guidance_1 = periodic_day_1 * periodic_week_1
        periodic_guidance_2 = periodic_day_2 * periodic_week_2

        return combined_temporal_embedding, periodic_guidance_1, periodic_guidance_2


class ESCUGraphLearning(nn.Module):
    def __init__(self, channels=128, num_nodes=170):
        super().__init__()
        self.memory_bank = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory_bank)
        self.graph_fusion = nn.Linear(2, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _normalize_graph(adj):
        identity = torch.eye(adj.shape[-1], device=adj.device)
        adj = adj + identity
        return adj / (adj.sum(-1, keepdim=True) + 1e-12)

    def forward(self, x, periodic_guidance_1=None, periodic_guidance_2=None):
        channels = x.shape[1]

        memory_graph = torch.softmax(
            F.relu(torch.einsum("bcnt,cm->bnm", x, self.memory_bank) / math.sqrt(channels)),
            dim=-1
        )

        feature_graph = torch.softmax(
            F.relu(torch.einsum("bcn,bcm->bnm", x.sum(-1), x.sum(-1)) / math.sqrt(channels)),
            dim=-1
        )

        event_graph = torch.softmax(
            self.graph_fusion(torch.stack([memory_graph, feature_graph], dim=-1)).squeeze(-1),
            dim=-1
        )

        if periodic_guidance_1 is not None and periodic_guidance_2 is not None:
            p1 = torch.tanh(periodic_guidance_1)
            p2 = torch.tanh(periodic_guidance_2)

            periodic_graph_in = self._normalize_graph(F.relu(torch.matmul(p1, p2.transpose(2, 1))))
            periodic_graph_out = self._normalize_graph(F.relu(torch.matmul(p2, p1.transpose(2, 1))))
            periodic_graph = 0.5 * (periodic_graph_in + periodic_graph_out)

            gamma = torch.clamp(self.gamma, 0.0, 1.0)
            event_graph = gamma * periodic_graph + (1.0 - gamma) * event_graph

        k = max(1, int(event_graph.shape[-1] * 0.8))
        _, index = torch.topk(event_graph, k=k, dim=-1)
        mask = torch.zeros_like(event_graph).scatter_(-1, index, 1.0)

        event_graph = event_graph * mask
        event_graph = event_graph / (event_graph.sum(-1, keepdim=True) + 1e-12)

        return event_graph


class ESCU(nn.Module):
    def __init__(self, channels=128, num_nodes=170, gating_embedding=None, dropout=0.1):
        super().__init__()
        self.linear_projection = nn.Conv2d(channels, channels, (1, 1))
        self.graph_learning = ESCUGraphLearning(channels, num_nodes)
        self.dagcl = DAGCL(channels=channels, pdim_ratio=0.5, diffusion_order=2, dropout=dropout)
        self.gating_embedding = gating_embedding

    def forward(self, x, periodic_guidance_1=None, periodic_guidance_2=None):
        residual = x
        x = self.linear_projection(x)
        event_graph = self.graph_learning(x, periodic_guidance_1, periodic_guidance_2)
        x = self.dagcl(x, event_graph)
        return x * torch.sigmoid(self.gating_embedding) + residual


class TSCUGraphLearning(nn.Module):
    def __init__(self, num_nodes=170):
        super().__init__()
        self.graph_fusion = nn.Linear(2, 1)
        self.adaptive_embedding = nn.Parameter(torch.randn(num_nodes, 10))
        nn.init.xavier_uniform_(self.adaptive_embedding)
        self.lambda_weight = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _normalize_graph(adj):
        identity = torch.eye(adj.shape[-1], device=adj.device)
        adj = adj + identity
        return adj / (adj.sum(-1, keepdim=True) + 1e-12)

    def forward(self, x, periodic_guidance_1=None, periodic_guidance_2=None):
        channels = x.shape[1]

        feature_graph = torch.softmax(
            F.relu(torch.einsum("bcn,bcm->bnm", x.sum(-1), x.sum(-1)) / math.sqrt(channels)),
            dim=-1
        )

        if periodic_guidance_1 is not None and periodic_guidance_2 is not None:
            p1 = torch.tanh(periodic_guidance_1)
            p2 = torch.tanh(periodic_guidance_2)

            periodic_graph_in = self._normalize_graph(F.relu(torch.matmul(p1, p2.transpose(2, 1))))
            periodic_graph_out = self._normalize_graph(F.relu(torch.matmul(p2, p1.transpose(2, 1))))
            periodic_graph = 0.5 * (periodic_graph_in + periodic_graph_out)

            lambda_weight = torch.clamp(self.lambda_weight, 0.0, 1.0)
            dynamic_graph = lambda_weight * periodic_graph + (1.0 - lambda_weight) * feature_graph
        else:
            dynamic_graph = feature_graph

        adaptive_graph = F.relu(torch.mm(self.adaptive_embedding, self.adaptive_embedding.t()))
        adaptive_graph = F.softmax(adaptive_graph, dim=1)
        adaptive_graph = adaptive_graph.unsqueeze(0).expand(x.shape[0], -1, -1)

        trend_graph = torch.softmax(
            self.graph_fusion(torch.stack([dynamic_graph, adaptive_graph], dim=-1)).squeeze(-1),
            dim=-1
        )

        k = max(1, int(trend_graph.shape[-1] * 0.8))
        _, index = torch.topk(trend_graph, k=k, dim=-1)
        mask = torch.zeros_like(trend_graph).scatter_(-1, index, 1.0)

        trend_graph = trend_graph * mask
        trend_graph = trend_graph / (trend_graph.sum(-1, keepdim=True) + 1e-12)

        return trend_graph


class TSCU(nn.Module):
    def __init__(self, channels=128, num_nodes=170, gating_embedding=None, dropout=0.1):
        super().__init__()
        self.linear_projection = nn.Conv2d(channels, channels, (1, 1))
        self.graph_learning = TSCUGraphLearning(num_nodes)
        self.dagcl = DAGCL(channels=channels, pdim_ratio=0.5, diffusion_order=2, dropout=dropout)
        self.gating_embedding = gating_embedding

    def forward(self, x, periodic_guidance_1=None, periodic_guidance_2=None):
        residual = x
        x = self.linear_projection(x)
        trend_graph = self.graph_learning(x, periodic_guidance_1, periodic_guidance_2)
        x = self.dagcl(x, trend_graph)
        return x * torch.sigmoid(self.gating_embedding) + residual


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :, :-self.chomp_size].contiguous()


class CausalTemporalConvolution(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super().__init__()
        layers = []

        for level in range(levels):
            dilation = 2 ** level
            padding = (kernel_size - 1) * dilation

            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        (1, kernel_size),
                        dilation=(1, dilation),
                        padding=(0, padding)
                    ),
                    Chomp1d(padding),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PGHANet(nn.Module):
    def __init__(self, device, input_dim, num_nodes, channels, granularity, dropout=0.1):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        self.heads = 8

        self.itpe = ITPE(granularity, channels)
        self.input_projection = nn.Conv2d(in_channels=input_dim, out_channels=channels, kernel_size=(1, 1))

        self.trend_temporal_encoder = nn.ModuleList(
            [
                TemporalSelfAttentionLayer(channels * 2, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(3)
            ]
        )

        self.event_temporal_encoder = CausalTemporalConvolution(channels * 2)

        self.trend_gating_embedding = nn.Parameter(torch.randn(channels * 2, num_nodes, 12))
        self.event_gating_embedding = nn.Parameter(torch.randn(channels * 2, num_nodes, 12))

        self.tscu = TSCU(
            channels=channels * 2,
            num_nodes=num_nodes,
            gating_embedding=self.trend_gating_embedding,
            dropout=dropout
        )

        self.escu = ESCU(
            channels=channels * 2,
            num_nodes=num_nodes,
            gating_embedding=self.event_gating_embedding,
            dropout=dropout
        )

        self.glu = GLU(channels * 2, dropout)

        self.prediction_layer = nn.Conv2d(
            channels * 2,
            self.output_len,
            kernel_size=(1, self.output_len)
        )

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputxl, inputxh):
        trend_temporal_embedding, trend_periodic_1, trend_periodic_2 = self.itpe(
            inputxl.permute(0, 3, 2, 1),
            return_periodic_guidance=True
        )

        event_temporal_embedding, event_periodic_1, event_periodic_2 = self.itpe(
            inputxh.permute(0, 3, 2, 1),
            return_periodic_guidance=True
        )

        trend_input = torch.cat(
            [self.input_projection(inputxl), trend_temporal_embedding],
            dim=1
        )

        event_input = torch.cat(
            [self.input_projection(inputxh), event_temporal_embedding],
            dim=1
        )

        trend_hidden = trend_input.permute(0, 3, 2, 1)

        for attention_layer in self.trend_temporal_encoder:
            trend_hidden = attention_layer(trend_hidden, attention_dim=1)

        trend_hidden = trend_hidden.permute(0, 3, 2, 1).contiguous()

        trend_periodic_1 = trend_periodic_1.mean(dim=1)
        trend_periodic_2 = trend_periodic_2.mean(dim=1)

        trend_hidden = self.tscu(
            trend_hidden,
            trend_periodic_1,
            trend_periodic_2
        )

        trend_hidden = self.glu(trend_hidden) + trend_hidden

        event_hidden = self.event_temporal_encoder(event_input)

        event_periodic_1 = event_periodic_1.mean(dim=1)
        event_periodic_2 = event_periodic_2.mean(dim=1)

        event_hidden = self.escu(
            event_hidden,
            event_periodic_1,
            event_periodic_2
        )

        fused_hidden = trend_hidden + event_hidden

        return self.prediction_layer(F.relu(fused_hidden))
