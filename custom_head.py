"""
custom_head.py
=================
Два GeGLU-блока (h = Linear_gate(x) * gelu(Linear_up(x)) -> Linear_down(h),
плюс residual + LayerNorm) поверх замороженного e5 - "голова", которую реально
дообучаем. Один линейный слой слишком слабый, чтобы чему-то научиться при
полностью замороженном энкодере - GeGLU даёт нелинейность и gating.

Файл должен быть отдельным импортируемым .py (не кодом внутри ноутбука),
потому что sentence-transformers сохраняет/загружает кастомные модули по
импортируемому пути класса (см. modules.json внутри сохранённой модели).
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
    """Совместим с pipeline sentence-transformers: forward принимает и
    возвращает dict с ключом 'sentence_embedding'."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.block1 = GeGLUBlock(dim, hidden_dim)
        self.block2 = GeGLUBlock(dim, hidden_dim)

    def forward(self, features: dict) -> dict:
        x = features["sentence_embedding"]
        x = self.block1(x)
        x = self.block2(x)
        features["sentence_embedding"] = x
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def get_config_dict(self) -> dict:
        return {"dim": self.dim, "hidden_dim": self.hidden_dim}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f)
        if safe_serialization:
            save_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path: str) -> "DoubleGeGLUHead":
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        safetensors_path = os.path.join(input_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model
