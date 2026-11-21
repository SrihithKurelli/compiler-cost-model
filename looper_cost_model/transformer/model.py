
import torch
import torch.nn as nn


class TransformationTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        feedforward_dim=256,
        dropout=0.1,
        max_sequence_length=15,
        num_transform_types=9,
        max_loop_levels=3,
        max_parameters=3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Transformation types:
        # 0 = padding, 1-7 = transformations, 8 = CLS.
        self.type_embedding = nn.Embedding(
            num_transform_types,
            hidden_dim,
            padding_idx=0,
        )

        # Learned representation for the CLS token.
        self.cls_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        # Numerical feature projections.
        self.loop_projection = nn.Sequential(
            nn.Linear(max_loop_levels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.parameter_projection = nn.Sequential(
            nn.Linear(max_parameters, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Embeddings for the number of loop levels and parameters.
        self.loop_count_embedding = nn.Embedding(
            max_loop_levels + 1, hidden_dim
        )

        self.parameter_count_embedding = nn.Embedding(
            max_parameters + 1, hidden_dim
        )

        self.computation_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length, hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Program context branch.
        self.context_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Fuse schedule representation and program context.
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, batch):
        type_ids = batch["type_ids"]
        loop_levels = batch["loop_levels"]
        loop_counts = batch["loop_counts"]
        parameters = batch["parameters"]
        parameter_counts = batch["parameter_counts"]
        computation_counts = batch["computation_counts"]
        padding_mask = batch["padding_mask"]
        context = batch["context"]

        batch_size, sequence_length = type_ids.shape

        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum {self.position_embedding.num_embeddings}"
            )

        # Normalize loop indices to approximately [0, 1].
        loop_levels = loop_levels / 4.0

        # Build transformation token representations.
        x = self.type_embedding(type_ids)

        x = x + self.loop_projection(loop_levels)

        x = x + self.loop_count_embedding(loop_counts)

        x = x + self.parameter_projection(parameters)

        x = x + self.parameter_count_embedding(parameter_counts)

        x = x + self.computation_projection(
            computation_counts.unsqueeze(-1)
        )

        # Add positional information.
        positions = torch.arange(
            sequence_length,
            device=type_ids.device,
        ).unsqueeze(0)

        x = x + self.position_embedding(positions)

        # Add a learned CLS representation at position zero.
        cls = x[:, :1, :] + self.cls_embedding

        x = torch.cat([cls, x[:, 1:, :]], dim=1)

        # Encode the ordered transformation sequence.
        encoded = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # CLS representation summarises the schedule.
        schedule_representation = encoded[:, 0, :]

        # Encode program-level context.
        context_representation = self.context_encoder(context)

        # Fuse both representations.
        combined = torch.cat(
            [
                schedule_representation,
                context_representation,
            ],
            dim=1,
        )

        prediction = self.regression_head(combined)

        return prediction.squeeze(-1)
