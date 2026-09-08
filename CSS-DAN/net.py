import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from timm.models.layers import trunc_normal_, DropPath
import math

# --- Static Hyperparameters ---
DIMS = [96, 192, 384]
DEPTHS = [3, 3, 9]
DROP_PATH_RATE = 0.1
LAYER_SCALE_INIT_VALUE = 1e-6
NUM_BANDS = 5
NUM_CHANNELS = 30
GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, GAT_N_HEADS, GAT_DROPOUT, GAT_ALPHA = 128, 64, 5, 0.1, 0.1
FUSION_DIM, FUSION_N_HEADS = 256, 8
SEQUENCE_LENGTH = 10
TRANSFORMER_D_MODEL = 128
TRANSFORMER_N_HEADS = 8


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Calculates the Gaussian kernel matrix for MMD loss."""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Calculates MMD loss using an RBF kernel."""
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class GradientReverseLayer(Function):
    """Gradient reversal layer for DANN."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features, self.out_features, self.dropout, self.alpha, self.concat = in_features, out_features, dropout, alpha, concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)
        N = h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(h.size(0), N * N, -1), h.repeat(1, N, 1)], dim=-1).view(h.size(0),
                                                                                                            N, N,
                                                                                                            2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.relu(h_prime) if self.concat else h_prime


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions): self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight, self.bias, self.eps, self.data_format, self.normalized_shape = nn.Parameter(
            torch.ones(normalized_shape)), nn.Parameter(torch.zeros(normalized_shape)), eps, data_format, (
            normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u, s = x.mean(1, keepdim=True), (x - x.mean(1, keepdim=True)).pow(2).mean(1, keepdim=True)
            return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input_tensor + self.drop_path(x)
        return x


def normalize_A_batch(A):
    """Normalizes the adjacency matrix for a batch."""
    A = F.relu(A)
    d = torch.sum(A, dim=-1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    D = torch.diag_embed(d_inv_sqrt)
    return torch.bmm(torch.bmm(D, A), D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SharedExtractor(nn.Module):
    """
    Shared feature extractor with dynamic dimension handling
    """

    def __init__(self, in_chans=1, depths=DEPTHS, dims=DIMS, drop_path_rate=DROP_PATH_RATE,
                 layer_scale_init_value=LAYER_SCALE_INIT_VALUE):
        super().__init__()
        self.downsample_layers, self.stages = self._build_csfnet(in_chans, depths, dims, drop_path_rate,
                                                                 layer_scale_init_value)
        self.spatial_gat = GAT(n_feat=dims[-1], n_hid=GAT_HIDDEN_DIM, n_class=GAT_OUTPUT_DIM, dropout=GAT_DROPOUT,
                               alpha=GAT_ALPHA, n_heads=GAT_N_HEADS)
        self.base_adj = nn.Parameter(torch.randn(NUM_CHANNELS, NUM_CHANNELS))
        nn.init.xavier_uniform_(self.base_adj.data)

        # 动态计算输入维度
        self.window_projector = None
        self.transformer_input_dim = None

        self.pos_encoder = PositionalEncoding(d_model=TRANSFORMER_D_MODEL, max_len=SEQUENCE_LENGTH + 5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_N_HEADS,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_dropout = nn.Dropout(0.2)

        self.gating_mlp = None
        self.gating_mlp_input_dim = None

        self.spatial_projection = nn.Linear(NUM_CHANNELS * GAT_OUTPUT_DIM, FUSION_DIM)
        self.temporal_projection = nn.Linear(SEQUENCE_LENGTH * TRANSFORMER_D_MODEL, FUSION_DIM)
        self.cnn_projection = nn.Linear(DIMS[-1], FUSION_DIM)
        self.cnn_cross_attention = nn.MultiheadAttention(
            embed_dim=FUSION_DIM, num_heads=FUSION_N_HEADS, batch_first=True
        )
        self.gat_cross_attention = nn.MultiheadAttention(
            embed_dim=FUSION_DIM, num_heads=FUSION_N_HEADS, batch_first=True
        )
        self.transformer_cross_attention = nn.MultiheadAttention(
            embed_dim=FUSION_DIM, num_heads=FUSION_N_HEADS, batch_first=True
        )

    def _build_csfnet(self, in_chans, depths, dims, drop_path_rate, layer_scale_init_value):
        downsample_layers, stem = nn.ModuleList(), nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(1, 2), stride=(1, 2)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        downsample_layers.append(stem)
        downsample_layers.append(nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                               nn.Conv2d(dims[0], dims[1], kernel_size=(1, 2), stride=(1, 2))))
        downsample_layers.append(nn.Sequential(LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                               nn.Conv2d(dims[1], dims[2], kernel_size=1, stride=1)))
        stages, dp_rates, cur = nn.ModuleList(), [x.item() for x in
                                                  torch.linspace(0, drop_path_rate, sum(depths))], 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j
                  in range(depths[i])])
            stages.append(stage)
            cur += depths[i]
        return downsample_layers, stages

    def _init_dynamic_layers(self, x):

        x_sequence_flat = x.view(x.size(0), SEQUENCE_LENGTH, -1)
        actual_feature_dim = x_sequence_flat.size(-1)

        if self.window_projector is None or self.transformer_input_dim != actual_feature_dim:
            self.transformer_input_dim = actual_feature_dim
            self.window_projector = nn.Linear(actual_feature_dim, TRANSFORMER_D_MODEL).to(x.device)

        transformer_output_dim = SEQUENCE_LENGTH * TRANSFORMER_D_MODEL
        if self.gating_mlp is None or self.gating_mlp_input_dim != transformer_output_dim:
            self.gating_mlp_input_dim = transformer_output_dim
            self.gating_mlp = nn.Sequential(
                nn.Linear(transformer_output_dim, 128), nn.ReLU(),
                nn.Linear(128, NUM_CHANNELS * NUM_CHANNELS),
                nn.Sigmoid()
            ).to(x.device)

    def forward(self, x):
        self._init_dynamic_layers(x)

        x_single_window = x[:, -1, :, :]
        x_for_cnn = x_single_window.permute(0, 2, 1)
        x_cnn = self.downsample_layers[0](x_for_cnn.unsqueeze(1))
        x_cnn = self.stages[0](x_cnn)
        x_cnn = self.downsample_layers[1](x_cnn)
        x_cnn = self.stages[1](x_cnn)
        x_cnn = self.downsample_layers[2](x_cnn)
        x_cnn = self.stages[2](x_cnn)
        residual_vector = x_cnn.mean(dim=[-2, -1])
        node_features = x_cnn.squeeze(-1).permute(0, 2, 1)

        x_sequence_flat = x.view(x.size(0), SEQUENCE_LENGTH, -1)
        transformer_input = self.window_projector(x_sequence_flat)
        transformer_input_pos = transformer_input.permute(1, 0, 2)
        transformer_input_pos = self.pos_encoder(transformer_input_pos)
        transformer_input_pos = transformer_input_pos.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(transformer_input_pos)
        transformer_output = self.transformer_dropout(transformer_output)
        temporal_context_vector = transformer_output.contiguous().view(x.size(0), -1)

        gating_matrix = self.gating_mlp(temporal_context_vector).view(-1, NUM_CHANNELS, NUM_CHANNELS)
        dynamic_adj = self.base_adj.unsqueeze(0) * gating_matrix
        dynamic_adj_normalized = normalize_A_batch(dynamic_adj)
        graph_features = self.spatial_gat(node_features, dynamic_adj_normalized)
        graph_features_flat = graph_features.view(graph_features.size(0), -1)

        proj_cnn = self.cnn_projection(residual_vector)
        proj_gat = self.spatial_projection(graph_features_flat)
        proj_transformer = self.temporal_projection(temporal_context_vector)

        q_cnn = proj_cnn.unsqueeze(1)
        q_gat = proj_gat.unsqueeze(1)
        q_transformer = proj_transformer.unsqueeze(1)

        kv_gat_transformer = torch.cat([q_gat, q_transformer], dim=1)
        cnn_context, _ = self.cnn_cross_attention(query=q_cnn, key=kv_gat_transformer, value=kv_gat_transformer)

        kv_cnn_transformer = torch.cat([q_cnn, q_transformer], dim=1)
        gat_context, _ = self.gat_cross_attention(query=q_gat, key=kv_cnn_transformer, value=kv_cnn_transformer)

        kv_cnn_gat = torch.cat([q_cnn, q_gat], dim=1)
        transformer_context, _ = self.transformer_cross_attention(query=q_transformer, key=kv_cnn_gat, value=kv_cnn_gat)

        final_features = torch.cat(
            [cnn_context.squeeze(1), gat_context.squeeze(1), transformer_context.squeeze(1)], dim=1
        )
        return final_features


class SpecificExtractor(nn.Module):


    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, x):
        if x.size(0) == 1:
            x = F.relu(self.mlp_layers[0](x))
            x = self.mlp_layers[2](x)  # Dropout
            x = F.relu(self.mlp_layers[3](x))
            x = self.mlp_layers[5](x)  # Dropout
            x = self.mlp_layers[6](x)
            return x
        else:
            x = self.mlp_layers[0](x)
            x = self.layer_norm1(x)
            x = F.relu(x)
            x = self.mlp_layers[2](x)
            x = self.mlp_layers[3](x)
            x = self.layer_norm2(x)
            x = F.relu(x)
            x = self.mlp_layers[5](x)
            x = self.mlp_layers[6](x)
            return x


class UnifiedDANN(nn.Module):
    def __init__(self, all_subject_ids, num_classes=1):
        super().__init__()
        self.shared_extractor = SharedExtractor()

        self.shared_output_dim = None

        alignment_dim = 256
        self.align_proj_shared = None
        self.align_proj_specific = nn.Linear(128, alignment_dim)

        self.specific_extractors = nn.ModuleDict()
        self.all_subject_ids = all_subject_ids
        self.specific_input_dim = None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(alignment_dim * 2, 512),
            nn.ReLU(), nn.Dropout(0.5)
        )

        fused_dim = 512
        self.label_predictor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(fused_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True), nn.Linear(100, 2), nn.LogSoftmax(dim=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _init_dynamic_layers(self, shared_features):
        if self.shared_output_dim is None:
            self.shared_output_dim = shared_features.size(-1)
            self.align_proj_shared = nn.Linear(self.shared_output_dim, 256).to(shared_features.device)

            if not self.specific_extractors:
                for sid in self.all_subject_ids:
                    self.specific_extractors[str(sid)] = SpecificExtractor(
                        input_dim=self.shared_output_dim,
                        output_dim=128
                    ).to(shared_features.device)

    def forward(self, x, subject_ids, alpha):
        # 1. Extract shared features
        shared_features = self.shared_extractor(x)

        self._init_dynamic_layers(shared_features)

        specific_features_list = []

        unique_subjects = torch.unique(subject_ids)
        subject_groups = {}

        for i, sid in enumerate(subject_ids):
            sid_str = str(sid.item())
            if sid_str not in subject_groups:
                subject_groups[sid_str] = []
            subject_groups[sid_str].append(i)

        for sid_str, indices in subject_groups.items():
            if len(indices) > 1:
                group_features = shared_features[indices]
                specific_features = self.specific_extractors[sid_str](group_features)
                specific_features_list.append(specific_features)
            else:
                single_feature = shared_features[indices[0]].unsqueeze(0)
                specific_features = self.specific_extractors[sid_str](single_feature)
                specific_features_list.append(specific_features)

        specific_features_batch = torch.cat(specific_features_list, dim=0)

        # 3. Project to alignment space
        shared_proj = self.align_proj_shared(shared_features)
        specific_proj = self.align_proj_specific(specific_features_batch)

        # 4. Calculate alignment loss using MMD loss
        loss_alignment = mmd_rbf(shared_proj, specific_proj)

        # 5. Concatenate and fuse the aligned features
        fused_features = self.fusion_mlp(torch.cat([shared_proj, specific_proj], dim=1))

        # 6. Prediction
        label_output = self.label_predictor(fused_features)
        reverse_features = GradientReverseLayer.apply(fused_features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output, shared_features, loss_alignment