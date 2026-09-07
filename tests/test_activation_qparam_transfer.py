import torch

from pytorchocr.quantization import (
    activation_qparam_sites,
    transfer_activation_qparams_by_site,
)


class DummyFakeQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))

    def forward(self, x):
        return x


class ReluObserved(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_post_process_0 = DummyFakeQuant()

    def forward(self, x):
        return self.activation_post_process_0(torch.relu(x))


class SigmoidObserved(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_post_process_0 = DummyFakeQuant()

    def forward(self, x):
        return self.activation_post_process_0(torch.sigmoid(x))


class StaticObserved(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.activation_post_process_0 = DummyFakeQuant()

    def forward(self, x):
        return x + self.activation_post_process_0(self.bias)


def _trace(module):
    class ObserverLeafTracer(torch.fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if isinstance(module, DummyFakeQuant):
                return True
            return super().is_leaf_module(module, module_qualified_name)

    tracer = ObserverLeafTracer()
    graph = tracer.trace(module)
    return torch.fx.GraphModule(module, graph)


def test_transfer_activation_qparams_matches_semantic_site():
    source = _trace(ReluObserved())
    target = _trace(ReluObserved())
    source_state = source.state_dict()
    source_state["activation_post_process_0.scale"] = torch.tensor([0.25])
    source_state["activation_post_process_0.zero_point"] = torch.tensor(
        [7], dtype=torch.int32
    )

    new_state, report = transfer_activation_qparams_by_site(
        source, source_state, target
    )

    assert report.copied == 2
    assert torch.equal(
        new_state["activation_post_process_0.scale"], torch.tensor([0.25])
    )
    assert torch.equal(
        new_state["activation_post_process_0.zero_point"],
        torch.tensor([7], dtype=torch.int32),
    )


def test_transfer_activation_qparams_rejects_same_name_different_semantics():
    source = _trace(ReluObserved())
    target = _trace(SigmoidObserved())
    source_state = source.state_dict()
    source_state["activation_post_process_0.scale"] = torch.tensor([0.25])

    new_state, report = transfer_activation_qparams_by_site(
        source, source_state, target
    )

    assert report.copied == 0
    assert torch.equal(
        new_state["activation_post_process_0.scale"],
        target.state_dict()["activation_post_process_0.scale"],
    )


def test_activation_qparam_sites_skip_static_get_attr_by_default():
    traced = _trace(StaticObserved())

    assert activation_qparam_sites(traced) == {}
    assert activation_qparam_sites(traced, include_get_attr=True)
