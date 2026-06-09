from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RerankerOutput:
    full_logits: torch.Tensor
    candidate_logits: torch.Tensor
    candidate_idx: torch.Tensor
    stage1_candidate_logits: torch.Tensor


class TwoStageCandidateReranker(nn.Module):
    """Standalone Top-K reranker trained on frozen Stage-1 beam predictions."""

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 64,
        topk: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        embed_dropout: float = 0.1,
        restrict_to_candidates: bool = True,
    ):
        super().__init__()
        if topk < 3:
            raise ValueError("topk must be >= 3")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if embed_dropout < 0 or embed_dropout >= 1:
            raise ValueError("embed_dropout must be in [0, 1)")

        self.num_classes = int(num_classes)
        self.topk = min(int(topk), self.num_classes)
        self.restrict_to_candidates = bool(restrict_to_candidates)
        self.embed_dropout = float(embed_dropout)

        self.context_norm = nn.LayerNorm(d_model)
        self.context_score = nn.Linear(d_model, 1)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.beam_embed = nn.Embedding(self.num_classes, d_model)
        self.logit_norm = nn.LayerNorm(self.topk)

        candidate_dim = d_model * 2 + 6
        self.input_proj = nn.Sequential(
            nn.LayerNorm(candidate_dim),
            nn.Linear(candidate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_scale = nn.Parameter(torch.tensor(0.20))
        self.register_buffer(
            "beam_positions",
            torch.linspace(-1.0, 1.0, steps=self.num_classes).view(1, self.num_classes),
            persistent=False,
        )

        final_linear = self.score[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def summarize_context(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.context_score(self.context_norm(fused_tokens)), dim=1)
        attn_pool = (fused_tokens * weights).sum(dim=1)
        mean_pool = fused_tokens.mean(dim=1)
        max_pool = fused_tokens.amax(dim=1)
        return self.context_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=1))

    def forward(self, stage1_logits: torch.Tensor, fused_tokens: torch.Tensor) -> RerankerOutput:
        candidate_logits, candidate_idx = stage1_logits.detach().topk(self.topk, dim=1)
        context = self.summarize_context(fused_tokens)

        normalized_logits = self.logit_norm(candidate_logits).unsqueeze(-1)
        candidate_probs = torch.softmax(candidate_logits, dim=1).unsqueeze(-1)
        rank_pos = torch.linspace(
            0.0,
            1.0,
            steps=self.topk,
            device=stage1_logits.device,
            dtype=stage1_logits.dtype,
        ).view(1, self.topk, 1).expand(stage1_logits.size(0), -1, -1)
        beam_pos = self.beam_positions.to(dtype=stage1_logits.dtype).expand(stage1_logits.size(0), -1)
        candidate_pos = beam_pos.gather(1, candidate_idx).unsqueeze(-1)
        top1_pos = candidate_pos[:, :1, :]
        rel_pos = candidate_pos - top1_pos
        logit_gap = (candidate_logits - candidate_logits[:, :1]).unsqueeze(-1)

        beam_features = self.beam_embed(candidate_idx)
        if self.embed_dropout > 0:
            beam_features = F.dropout(beam_features, p=self.embed_dropout, training=self.training)

        candidate_features = torch.cat(
            [
                beam_features,
                context.unsqueeze(1).expand(-1, self.topk, -1),
                normalized_logits,
                candidate_probs,
                rank_pos,
                candidate_pos,
                rel_pos,
                logit_gap,
            ],
            dim=-1,
        )
        encoded = self.candidate_encoder(self.input_proj(candidate_features))
        delta = self.score(encoded).squeeze(-1)
        delta = delta - delta.mean(dim=1, keepdim=True)
        reranked_candidate_logits = candidate_logits + self.delta_scale * delta

        if self.restrict_to_candidates:
            full_logits = stage1_logits.new_full(stage1_logits.shape, -1e4)
        else:
            full_logits = stage1_logits.clone()
        full_logits = full_logits.scatter(1, candidate_idx, reranked_candidate_logits)

        return RerankerOutput(
            full_logits=full_logits,
            candidate_logits=reranked_candidate_logits,
            candidate_idx=candidate_idx,
            stage1_candidate_logits=candidate_logits,
        )
