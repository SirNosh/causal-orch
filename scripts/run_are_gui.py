"""Launch Meta ARE's native GUI with the pinned causal agent and Gaia2 artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
GUI_SOURCE = ROOT / "artifacts" / "are-gui-source"
GUI_DATASET = ROOT / "artifacts" / "are-gui-dataset"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_smoke import SmokeExecuteSchedule, pinned_gaia2_factory


def _prepare_dataset() -> Path:
    manifest = json.loads(
        (ROOT / "configs" / "gaia2_manifest.json").read_text(encoding="utf-8")
    )
    scenario_id = manifest["scenario_ids"][0]
    source = ROOT / manifest["scenario_files"][scenario_id]
    if not source.is_file():
        raise RuntimeError(
            "pinned scenario is missing; run: uv run --with datasets scripts/fetch_pinned_gaia2.py"
        )
    target_dir = GUI_DATASET / manifest["selection"]["capability"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    return GUI_DATASET


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    gui_root = GUI_SOURCE / "are"
    if not (gui_root / "simulation" / "gui" / "client" / "build" / "index.html").is_file():
        raise RuntimeError("native ARE GUI is not built; run: uv run python scripts/setup_are_gui.py")

    from are.simulation.gui.server import server as server_module
    from are.simulation.gui.server.graphql.query import Query
    from are.simulation.gui.server.graphql.schema import Mutation
    from are.simulation.gui.server.graphql.subscription import Subscription
    from are.simulation.notification_system import VerboseNotificationSystem
    from causal_orch.runner.agent_builder import CausalAgentBuilder
    from causal_orch.runner.config_builder import CausalAgentConfigBuilder

    server_module.ARE_SIMULATION_ROOT = gui_root
    harness = pinned_gaia2_factory()
    experiment = harness.agent_builder.experiment_config
    forced_experiment = replace(
        experiment,
        intervention_schedule=SmokeExecuteSchedule(experiment.intervention_block),
    )
    server = server_module.ARESimulationGuiServer(
        hostname=args.host,
        port=args.port,
        certfile="",
        keyfile="",
        debug=False,
        scenario_id=None,
        agent="causal_orchestrator",
        model=forced_experiment.model_config.model_slug,
        provider=None,
        endpoint=forced_experiment.model_config.endpoint,
        agent_config_builder=CausalAgentConfigBuilder(forced_experiment),
        agent_builder=CausalAgentBuilder(forced_experiment),
        default_ui_view="SCENARIOS",
        dataset_path=str(_prepare_dataset()),
        notification_system_builder=VerboseNotificationSystem,
    )
    Query.server = server
    Mutation.server = server
    Subscription.server = server
    print(f"Meta ARE GUI: http://{args.host}:{args.port}")
    print("Load source Local, capability Search, scenario scenario_universe_28_2nr5po.json")
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
