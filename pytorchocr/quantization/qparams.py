"""Safe activation qparam migration helpers for PT2E QAT graphs.

Activation qparams are tied to the prepared FX graph.  Directly copying every
``activation_post_process_*`` entry by name across non-reparameterized and
reparameterized graphs can silently load but destroy fake-quant semantics.  The
helpers here only match observer sites by their producer semantics and report
what was copied or skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.fx import Node


QPARAM_FIELDS = ("scale", "zero_point")


@dataclass(frozen=True)
class ActivationQparamTransferReport:
    source_sites: int
    target_sites: int
    matched_sites: int
    copied: int
    skipped_shape: int
    skipped_missing: int
    include_get_attr: bool

    def as_dict(self):
        return {
            "source_sites": self.source_sites,
            "target_sites": self.target_sites,
            "matched_sites": self.matched_sites,
            "copied": self.copied,
            "skipped_shape": self.skipped_shape,
            "skipped_missing": self.skipped_missing,
            "include_get_attr": self.include_get_attr,
        }


def _observer_producer_site(node, *, include_get_attr):
    producer = node.args[0] if node.args else None
    if not isinstance(producer, Node):
        return None
    if producer.op == "get_attr" and not include_get_attr:
        return None
    module_stack = producer.meta.get("nn_module_stack") or {}
    module_path = ""
    if module_stack:
        module_path = list(module_stack.values())[-1][0] or ""
    source_fn_stack = producer.meta.get("source_fn_stack") or ()
    return (
        module_path,
        producer.op,
        str(producer.target),
        repr(source_fn_stack),
    )


def activation_qparam_sites(prepared, *, include_get_attr=False):
    """Return a unique semantic-site -> observer-name map.

    Ambiguous sites are intentionally dropped.  A duplicate semantic key means
    the graph has multiple observer nodes that cannot be safely distinguished
    using stable metadata alone.
    """
    sites = {}
    duplicates = set()
    for node in prepared.graph.nodes:
        if node.op != "call_module" or not str(node.target).startswith(
            "activation_post_process"
        ):
            continue
        site = _observer_producer_site(node, include_get_attr=include_get_attr)
        if site is None:
            continue
        if site in sites:
            duplicates.add(site)
            continue
        sites[site] = str(node.target)
    for site in duplicates:
        sites.pop(site, None)
    return sites


def transfer_activation_qparams_by_site(
    source_prepared,
    source_state,
    target_prepared,
    *,
    target_state=None,
    fields: Iterable[str] = QPARAM_FIELDS,
    include_get_attr=False,
):
    """Copy activation qparams from ``source_state`` into a target state copy.

    The match key is the observer producer's semantic site, not the unstable
    ``activation_post_process_N`` name.  Only ``scale`` and ``zero_point`` are
    copied by default; observer min/max statistics are deliberately not copied
    because they are not the active fake-quant qparams after observers are
    disabled.

    Returns ``(new_state, report)`` and never mutates ``target_prepared``.
    """
    allowed_fields = tuple(fields)
    for field in allowed_fields:
        if field not in QPARAM_FIELDS:
            raise ValueError(
                f"Unsupported activation qparam field {field!r}; "
                f"expected one of {QPARAM_FIELDS}."
            )
    source_sites = activation_qparam_sites(
        source_prepared, include_get_attr=include_get_attr
    )
    target_sites = activation_qparam_sites(
        target_prepared, include_get_attr=include_get_attr
    )
    new_state = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in (target_state or target_prepared.state_dict()).items()
    }
    copied = 0
    skipped_shape = 0
    skipped_missing = 0
    matched = sorted(set(source_sites).intersection(target_sites))
    for site in matched:
        source_observer = source_sites[site]
        target_observer = target_sites[site]
        for field in allowed_fields:
            source_key = f"{source_observer}.{field}"
            target_key = f"{target_observer}.{field}"
            if source_key not in source_state or target_key not in new_state:
                skipped_missing += 1
                continue
            source_value = source_state[source_key]
            target_value = new_state[target_key]
            if not torch.is_tensor(source_value) or not torch.is_tensor(target_value):
                skipped_missing += 1
                continue
            if source_value.shape != target_value.shape:
                skipped_shape += 1
                continue
            new_state[target_key] = source_value.detach().clone().to(target_value)
            copied += 1
    report = ActivationQparamTransferReport(
        source_sites=len(source_sites),
        target_sites=len(target_sites),
        matched_sites=len(matched),
        copied=copied,
        skipped_shape=skipped_shape,
        skipped_missing=skipped_missing,
        include_get_attr=include_get_attr,
    )
    return new_state, report
