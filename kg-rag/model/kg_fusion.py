import torch
import torch.nn as nn
from torch_geometric.nn import RGATConv


class KnowledgeFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gnn = RGATConv(hidden_size, hidden_size, num_relations=5)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, text_embeddings, kg_data):
        # 知识图谱编码
        kg_emb = self.gnn(kg_data.x, kg_data.edge_index, kg_data.edge_type)

        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=text_embeddings.transpose(0, 1),
            key=kg_emb.transpose(0, 1),
            value=kg_emb.transpose(0, 1)
        )
        return attn_out.transpose(0, 1)