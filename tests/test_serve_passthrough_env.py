# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`serve.passthrough_env`: variables the operator supplies at serve time, by name.

A server that talks to something beside it -- an object store, a database -- needs an
endpoint and credentials that belong to whoever runs it. ``serve.env`` is published with
the bundle, so it cannot carry them. These names are forwarded the way ``HF_TOKEN`` is:
``--env NAME`` with no value, so docker copies it from the caller's environment and it
never appears in ``--print`` output or ``ps``.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tt_kernel import container
from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.launchers import launcher_for

from test_container_manifest import BASE


def _wire(serve_over=None, profiles=None):
    raw = json.loads(json.dumps(BASE))
    raw["serve"] = {**raw.get("serve", {}), **(serve_over or {})}
    if profiles is not None:
        raw["serve_profiles"] = profiles
        raw["default_profile"] = profiles[0]["name"]
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m.to_wire(
        image_tag="tt-model/my-model:abc123",
        tt_metal_version="0.72.1",
        tt_kernel_version="0.1.0",
        hostname="h",
        created_at="2026-01-01T00:00:00+00:00",
    )


def _argv(m, passthrough_from, profile=None):
    p = m.container.resolve_profile(profile)
    launcher = launcher_for(m.container.kind)
    return container.compose_run(
        m, p, launcher.serve_argv(m, p), launcher.serve_env(m, p),
        hf_home_dir=Path("/h"), cache_dir=Path("/c"),
        weight_cache_dir=Path("/w"), tensor_cache_dir=Path("/t"),
        include_hf_token=False, passthrough_from=passthrough_from,
    )


def _env_flags(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]


def test_a_set_name_is_forwarded_by_name_only():
    m = _wire({"passthrough_env": ["STORE_ENDPOINT", "STORE_SECRET"]})
    argv = _argv(m, {"STORE_ENDPOINT": "http://10.0.0.5:9000", "STORE_SECRET": "s3cr3t"})

    flags = _env_flags(argv)
    assert "STORE_ENDPOINT" in flags and "STORE_SECRET" in flags
    joined = " ".join(argv)
    assert "s3cr3t" not in joined and "10.0.0.5" not in joined


def test_an_unset_name_is_not_forwarded():
    """Forwarding it would hand the server an empty string, which is not what 'the
    operator did not configure this' means to it."""
    m = _wire({"passthrough_env": ["STORE_ENDPOINT", "STORE_SECRET"]})
    flags = _env_flags(_argv(m, {"STORE_ENDPOINT": "http://x:9000"}))
    assert "STORE_ENDPOINT" in flags
    assert "STORE_SECRET" not in flags


def test_nothing_is_forwarded_that_was_not_declared():
    m = _wire({"passthrough_env": ["STORE_ENDPOINT"]})
    flags = _env_flags(_argv(m, {"STORE_ENDPOINT": "x", "SOMETHING_ELSE": "y"}))
    assert "SOMETHING_ELSE" not in flags


def test_a_value_the_package_sets_is_not_also_forwarded():
    """`serve.env` already sends KEY=VALUE; a bare `--env KEY` beside it would be a
    second, conflicting source for the same variable."""
    m = _wire({"env": {"STORE_ENDPOINT": "http://pinned:9000"},
               "passthrough_env": ["STORE_ENDPOINT"]})
    flags = _env_flags(_argv(m, {"STORE_ENDPOINT": "http://host:9000"}))
    assert "STORE_ENDPOINT=http://pinned:9000" in flags
    assert "STORE_ENDPOINT" not in flags


def test_the_declaration_travels_on_the_wire_and_merges_per_profile():
    m = _wire(
        {"passthrough_env": ["STORE_ENDPOINT"]},
        profiles=[
            {"name": "a", "hardware": "p150", "mesh_device": "P150",
             "max_num_seqs": 1, "block_size": 64},
            {"name": "b", "hardware": "p150", "mesh_device": "P150",
             "max_num_seqs": 1, "block_size": 64, "passthrough_env": ["OTHER"]},
        ],
    )
    reloaded = type(m).model_validate_json(m.model_dump_json())
    assert reloaded.container.resolve_profile("a").passthrough_env == ["STORE_ENDPOINT"]
    assert reloaded.container.resolve_profile("b").passthrough_env == ["OTHER"]


def test_without_a_declaration_the_argv_is_unchanged():
    m = _wire()
    assert _argv(m, {"STORE_ENDPOINT": "x"}) == _argv(m, {})


@pytest.mark.parametrize("bad", [["1ABC"], ["A-B"], ["A B"], ["X", "X"]])
def test_a_name_that_cannot_be_an_environment_variable_is_refused_at_load(bad):
    raw = json.loads(json.dumps(BASE))
    raw["serve"] = {**raw.get("serve", {}), "passthrough_env": bad}
    with pytest.raises(ValidationError, match="passthrough_env"):
        ContainerManifest.model_validate(raw)
