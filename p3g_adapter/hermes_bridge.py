"""Read-only bridge to the Hermes session gateway (Paket 3b) for AgentDojo.

Loads the *unmodified* files ``agent/session_gateway.py`` and
``agent/tool_dispatch_helpers.py`` (plus their two small helper imports) of the
Paket-3b commit ``c8341ea748`` (branch ``feat/paket3-gateway``) into this
process. The files are taken with ``git show <commit>:<path>`` from the worktree
``$P3G_HERMES_WORKTREE`` (a Hermes git checkout containing the session gateway) into ``vendor/hermes_<commit>/`` (gitignored): another
agent edits the worktree's working copy in parallel, so the working copy is
never imported. Nothing in the worktree is written.

Only the surroundings are replaced by small in-process stand-ins (no bytecode
is written, ``sys.dont_write_bytecode``):

* ``hermes_cli.config.load_config_readonly`` -> returns the switch block of
  the selected benchmark condition instead of reading the user's Hermes home.
* ``agent.session_gateway_axes.TOOL_AXES`` -> the AgentDojo tool mapping from
  :mod:`p3g_adapter.axes` (keys ``agentdojo.<tool>``) instead of the Hermes
  inventory, so the real rule code classifies AgentDojo tools.
* ``hermes_constants.get_hermes_dir`` -> a non-existent scratch path (only
  used for the delegation-cache path check, which never matches here).

The bridge must be configured once per process (one condition per process).
Condition ``mit_r4`` loads commit ``d8c3aafb22`` (refinement level r4,
``verfeinerung: r4``) instead; that commit additionally needs the 3c-3f
modules ``agent/session_gateway_{critic,egress,limits,refine,children}.py``
(exported and sha256-checked the same way, into ``vendor/hermes_d8c3aafb22``).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

WORKTREE = Path(os.environ.get("P3G_HERMES_WORKTREE", str(Path(__file__).resolve().parent.parent / "vendor" / "hermes-agent")))
COMMIT = os.environ.get("P3G_HERMES_COMMIT", "c8341ea748")
# Refinement levels r1..r4 of the session gateway.
COMMIT_R4 = "d8c3aafb22"
LOADED_FILES = ("agent/session_gateway.py", "agent/tool_dispatch_helpers.py")
SNAPSHOT_FILES = (
    *LOADED_FILES,
    "agent/message_metadata.py",
    "agent/tool_result_classification.py",
    "tools/threat_patterns.py",
)
# From 2a373b94a3 on, session_gateway.py imports the 3c/3d/3e modules at
# module level; d8c3aafb22 adds the refinement module (lazy imports).
EXTRA_FILES_R4 = (
    "agent/session_gateway_critic.py",
    "agent/session_gateway_egress.py",
    "agent/session_gateway_limits.py",
    "agent/session_gateway_refine.py",
    "agent/session_gateway_children.py",
)
VENDOR = Path(__file__).resolve().parent.parent / "vendor"

_STATE: Dict[str, Any] = {"switches": None, "gw": None, "helpers": None, "provenance": None, "commit": None}


def _same(a: str, b: str) -> bool:
    """Abbreviated or full hex commit ids name the same commit."""
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


_TREE_CACHE: Dict[str, tuple] = {}


def _gateway_modules(commit: str) -> tuple:
    """``agent/session_gateway_*.py`` of *commit* (except the axes table,
    which is always replaced by the AgentDojo mapping)."""
    if commit not in _TREE_CACHE:
        names = _git("ls-tree", "--name-only", f"{commit}", "agent/").splitlines()
        _TREE_CACHE[commit] = tuple(sorted(
            n for n in names
            if n.startswith("agent/session_gateway_") and n.endswith(".py") and n != "agent/session_gateway_axes.py"
        ))
    return _TREE_CACHE[commit]


def snapshot_files(commit: Optional[str] = None) -> tuple:
    """Files exported for *commit* (the r4 commit needs the 3c-3f modules;
    any other later commit, e.g. a Pruefstand gateway version, gets all its
    ``agent/session_gateway_*`` modules)."""
    c = commit or active_commit()
    if _same(c, COMMIT):
        return SNAPSHOT_FILES
    if _same(c, COMMIT_R4):
        return (*SNAPSHOT_FILES, *EXTRA_FILES_R4)
    return (*SNAPSHOT_FILES, *_gateway_modules(c))


def active_commit() -> str:
    return _STATE["commit"] or COMMIT


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(*args: str, binary: bool = False):
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(WORKTREE), *args], capture_output=True, check=True
    ).stdout
    return out if binary else out.decode().strip()


def full_commit(commit: Optional[str] = None) -> str:
    return _git("rev-parse", f"{commit or active_commit()}^{{commit}}")


def snapshot_dir(commit: Optional[str] = None) -> Path:
    return VENDOR / f"hermes_{commit or active_commit()}"


def ensure_snapshot(commit: Optional[str] = None) -> Dict[str, Any]:
    """Write the committed files to ``vendor/hermes_<commit>`` (idempotent) and
    return their provenance (blob sha256 == file sha256 is checked)."""
    commit = commit or active_commit()
    root = snapshot_dir(commit)
    files = {}
    for rel in snapshot_files(commit):
        blob = _git("show", f"{commit}:{rel}", binary=True)
        dst = root / rel
        if not dst.is_file() or dst.read_bytes() != blob:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob)
        files[rel] = _sha256_bytes(blob)
    for rel, digest in files.items():  # exported file == blob
        if _sha256_bytes((root / rel).read_bytes()) != digest:
            raise RuntimeError(f"snapshot {rel} differs from commit {commit}")
    return {"worktree": str(WORKTREE), "commit": full_commit(commit), "snapshot": str(root), "files_sha256": files}


def provenance() -> Dict[str, Any]:
    return ensure_snapshot()


def _install_stubs(switches: Mapping[str, Any], axes: Mapping[str, tuple]) -> None:
    sys.dont_write_bytecode = True  # never write __pycache__ into the worktree

    cfg_mod = types.ModuleType("hermes_cli.config")

    def load_config_readonly() -> Dict[str, Any]:
        return {"security": {"session_gateway": dict(_STATE["switches"] or {})}}

    cfg_mod.load_config_readonly = load_config_readonly  # type: ignore[attr-defined]
    cli_pkg = types.ModuleType("hermes_cli")
    cli_pkg.__path__ = []  # type: ignore[attr-defined]
    cli_pkg.config = cfg_mod  # type: ignore[attr-defined]

    const_mod = types.ModuleType("hermes_constants")

    def get_hermes_dir(*_a: Any, **_k: Any) -> Path:
        return Path("/nonexistent/p3g-agentdojo/hermes-home/cache/delegation")

    const_mod.get_hermes_dir = get_hermes_dir  # type: ignore[attr-defined]

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = [str(snapshot_dir() / "agent")]  # type: ignore[attr-defined]
    tools_pkg = types.ModuleType("tools")
    tools_pkg.__path__ = [str(snapshot_dir() / "tools")]  # type: ignore[attr-defined]

    # Pruefstand (critic): ``tools.registry.registry`` with the AgentDojo
    # function schemas (name, description, parameters) so the critic knows
    # argument names and, for v2, the trusted tool description.
    reg_mod = types.ModuleType("tools.registry")
    reg_mod.registry = _Registry()  # type: ignore[attr-defined]
    tools_pkg.registry = reg_mod  # type: ignore[attr-defined]

    axes_mod = types.ModuleType("agent.session_gateway_axes")
    axes_mod.AXIS_NAMES = ("egress", "irreversibel", "liest_sensible_daten", "taint_quelle")  # type: ignore[attr-defined]
    axes_mod.TOOL_AXES = dict(axes)  # type: ignore[attr-defined]
    axes_mod.AXES_SOURCE_SHA256 = "agentdojo-mapping"  # type: ignore[attr-defined]

    for name, mod in (
        ("hermes_cli", cli_pkg),
        ("hermes_cli.config", cfg_mod),
        ("hermes_constants", const_mod),
        ("agent", agent_pkg),
        ("tools", tools_pkg),
        ("agent.session_gateway_axes", axes_mod),
        ("tools.registry", reg_mod),
    ):
        existing = sys.modules.get(name)
        if existing is not None and existing is not mod and not getattr(existing, "_p3g_stub", False):
            raise RuntimeError(f"module {name!r} already imported from elsewhere: {existing!r}")
        mod._p3g_stub = True  # type: ignore[attr-defined]
        sys.modules[name] = mod


class _Registry:
    """Stand-in for ``tools.registry.registry`` (AgentDojo schemas, lazy)."""

    _schemas: Optional[Dict[str, Dict[str, Any]]] = None

    def _all(self) -> Dict[str, Dict[str, Any]]:
        if _Registry._schemas is None:
            from agentdojo.task_suite.load_suites import get_suite

            from p3g_adapter.axes import TOOL_MAP, key

            out: Dict[str, Dict[str, Any]] = {}
            for suite_name in TOOL_MAP:
                for fn in get_suite("v1.2.2", suite_name).tools:
                    out[key(fn.name)] = {"name": key(fn.name), "description": fn.description,
                                         "parameters": fn.parameters.model_json_schema()}
            _Registry._schemas = out
        return _Registry._schemas

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self._all().get(name)

    def get_entry(self, name: str) -> Optional[Dict[str, Any]]:
        return self._all().get(name)


def configure(switches: Mapping[str, Any], commit: Optional[str] = None) -> types.ModuleType:
    """Load the Hermes gateway of *commit* (once per process) and set the
    switches of this process. A second call with another commit raises (one
    commit per process: the modules live in ``sys.modules``)."""
    from p3g_adapter.axes import hermes_axes_table

    commit = commit or _STATE["commit"] or COMMIT
    if _STATE["gw"] is not None and commit != _STATE["commit"]:
        raise RuntimeError(f"bridge already loaded commit {_STATE['commit']}, cannot switch to {commit}")
    _STATE["switches"] = dict(switches)
    if _STATE["gw"] is None:
        _STATE["commit"] = commit
        prov = ensure_snapshot(commit)
        _install_stubs(switches, hermes_axes_table())
        _STATE["gw"] = importlib.import_module("agent.session_gateway")
        _STATE["helpers"] = importlib.import_module("agent.tool_dispatch_helpers")
        if "agent/session_gateway_refine.py" in snapshot_files(commit):
            importlib.import_module("agent.session_gateway_refine")
        if "agent/session_gateway_critic.py" in snapshot_files(commit):
            importlib.import_module("agent.session_gateway_critic")
        for rel in snapshot_files(commit):
            if not rel.startswith("agent/session_gateway") and rel not in LOADED_FILES:
                continue
            mod = sys.modules.get(rel[:-3].replace("/", "."))
            if mod is None:
                continue  # lazily imported module, checked by the snapshot itself
            mod_file = Path(mod.__file__).resolve()
            if mod_file != (snapshot_dir(commit) / rel).resolve():
                raise RuntimeError(f"loaded {mod_file}, expected {snapshot_dir(commit) / rel}")
            if _sha256_bytes(mod_file.read_bytes()) != prov["files_sha256"][rel]:
                raise RuntimeError(f"{rel} differs from commit {commit}")
        _STATE["provenance"] = prov
    return _STATE["gw"]


def loaded_modules_sha256() -> Dict[str, str]:
    """sha256 of every loaded ``agent.*``/``tools.*`` module file (for tests)."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and (name.startswith("agent.") or name.startswith("tools.")):
            out[name] = _sha256_bytes(Path(f).read_bytes())
    return out


def gateway() -> types.ModuleType:
    if _STATE["gw"] is None:
        raise RuntimeError("hermes_bridge.configure() not called")
    return _STATE["gw"]


def wrap_untrusted(source: str, content: str) -> str:
    """Hermes' own ``_maybe_wrap_untrusted(..., force=True)``."""
    if _STATE["helpers"] is None:
        raise RuntimeError("hermes_bridge.configure() not called")
    return _STATE["helpers"]._maybe_wrap_untrusted(source, content, force=True)


def loaded_provenance() -> Optional[Dict[str, Any]]:
    return _STATE["provenance"]
