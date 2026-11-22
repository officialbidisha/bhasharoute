import torch
from torch import nn

from bhasharoute.eval.latency import int8_layer_names, is_ffn_output


def _layer() -> nn.Module:
    m = nn.Module()
    m.attention = nn.Module()
    m.attention.self = nn.Module()
    m.attention.self.query = nn.Linear(4, 4)
    m.attention.output = nn.Module()
    m.attention.output.dense = nn.Linear(4, 4)
    m.intermediate = nn.Module()
    m.intermediate.dense = nn.Linear(4, 8)
    m.output = nn.Module()
    m.output.dense = nn.Linear(8, 4)
    return m


def test_is_ffn_output_distinguishes_attention_output():
    assert is_ffn_output("encoder.encoder.layer.3.output.dense")
    assert not is_ffn_output("encoder.encoder.layer.3.attention.output.dense")
    assert not is_ffn_output("encoder.encoder.layer.3.intermediate.dense")


def test_int8_skips_ffn_output_and_heads():
    net = nn.Module()
    net.encoder = nn.Module()
    net.encoder.layer = nn.ModuleList([_layer(), _layer()])
    net.heads = nn.Linear(4, 2)
    names = int8_layer_names(net)
    assert names == [f"encoder.layer.{i}.{p}" for i in range(2)
                     for p in ("attention.self.query", "attention.output.dense", "intermediate.dense")]
    assert all(isinstance(net.get_submodule(n), torch.nn.Linear) for n in names)
