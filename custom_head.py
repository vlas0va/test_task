"""Голова поверх замороженного энкодера: два GeGLU-блока с residual и LayerNorm.

Отдельный модуль, потому что sentence-transformers загружает кастомные модули
по пути класса (custom_head.DoubleGeGLUHead).
"""
import os
import json
import torch
from torch import nn
from safetensors.torch import save_file, load_file


class GeGLUBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.gate(x) * self.act(self.up(x))
        return self.norm(x + self.down(h))


class DoubleGeGLUHead(nn.Module):
    """Модуль sentence-transformers: принимает и возвращает dict с 'sentence_embedding'."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.block1 = GeGLUBlock(dim, hidden_dim)
        self.block2 = GeGLUBlock(dim, hidden_dim)

    def forward(self, features: dict) -> dict:
        features["sentence_embedding"] = self.block2(self.block1(features["sentence_embedding"]))
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def get_config_dict(self) -> dict:
        return {"dim": self.dim, "hidden_dim": self.hidden_dim}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f)
        save_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))

    @classmethod
    def load(cls, input_path: str) -> "DoubleGeGLUHead":
        with open(os.path.join(input_path, "config.json")) as f:
            model = cls(**json.load(f))
        path = os.path.join(input_path, "model.safetensors")
        if os.path.exists(path):
            state = load_file(path)
        else:
            state = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state)
        return model
