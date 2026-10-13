"""Shared command-line options and native dataset selection."""
import argparse

from project_paths import dataset_path, output_path


def parser_for(dataset):
    parser = argparse.ArgumentParser(description=f"Validate {dataset} preprocessing")
    parser.add_argument("--root", help="Dataset directory; overrides YAML")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--output", help="Output directory; overrides YAML")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--reference-agent", default=None)
    if dataset == "urbaning":
        parser.add_argument("--sequence")
        parser.add_argument("--indices", type=int, nargs="+")
        parser.add_argument("--show", action="store_true", help="Display plots interactively")
    elif dataset == "v2xpnp":
        parser.add_argument("--scenario", help="Scenario name or path relative to dataset root")
        parser.add_argument("--frame", help="Exact native frame ID, including leading zeroes")
        parser.add_argument("--vehicle-id")
        parser.add_argument("--infra-id")
        parser.add_argument("--split")
    return parser


def urbaning_adapter(args):
    from adapters.urbaning_adapter import UrbanIngAdapter
    root = dataset_path("urbaning", args.root, args.config)
    sequences = [args.sequence] if args.sequence else None
    adapter = UrbanIngAdapter(root, sequence_names=sequences,
                             reference_agent_id=args.reference_agent or "vehicle1")
    if args.reference_agent is None:
        sequence_name, frame_index = adapter._samples[args.index]
        frame = adapter._get_sequence(sequence_name)[frame_index]
        vehicles = sorted(key for key, vehicle in frame.vehicles.items()
                          if vehicle.state is not None)
        agents = vehicles or sorted(frame.infrastructures)
        if not agents:
            raise ValueError("Selected UrbanIng frame has no usable reference agent")
        adapter.reference_agent_id = agents[0]
    return adapter


def select_v2xpnp(args):
    """Select a complete vehicle/infrastructure pair by native metadata flags."""
    from adapters.v2xpnp_adapter import find_scenarios, is_agent_directory, read_yaml
    root = dataset_path("v2xpnp", args.root, args.config)
    if args.split:
        root = root / args.split
        if not root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {root}")
    scenarios = find_scenarios(root)
    if args.scenario:
        candidate = root / args.scenario
        scenarios = [candidate] if candidate.is_dir() else [
            path for path in scenarios if path.name == args.scenario]
        if len(scenarios) != 1:
            raise ValueError(f"Scenario missing or ambiguous: {args.scenario}")
    if args.index < 0:
        raise ValueError("Alignment --index must be nonnegative")
    if args.vehicle_id and args.reference_agent and args.vehicle_id != args.reference_agent:
        raise ValueError("--vehicle-id and --reference-agent disagree")
    vehicle_id = args.vehicle_id or args.reference_agent
    complete_index = 0
    for scenario in scenarios:
        agents = sorted((path for path in scenario.iterdir() if is_agent_directory(path)),
                        key=lambda path: int(path.name))
        frames = [args.frame] if args.frame else sorted({
            path.stem for agent in agents for path in agent.glob("*.yaml")
            if path.stem.isdigit()})
        for frame in frames:
            by_type = {False: [], True: []}
            for agent in agents:
                path = agent / f"{frame}.yaml"
                if not path.is_file() or not (agent / f"{frame}.bin").is_file():
                    continue
                metadata = read_yaml(path)
                flag = metadata.get("infra")
                if not isinstance(flag, bool):
                    raise ValueError(f"Missing/invalid infra flag: {path}")
                by_type[flag].append(agent.name)
            vehicles = [vehicle_id] if vehicle_id in by_type[False] else (
                [] if vehicle_id else by_type[False])
            infrastructures = [args.infra_id] if args.infra_id in by_type[True] else (
                [] if args.infra_id else by_type[True])
            if vehicles and infrastructures:
                if complete_index == args.index:
                    return scenario, frame, vehicles[0], infrastructures[0]
                complete_index += 1
    raise ValueError("No complete vehicle/infrastructure pair matches the requested selection")
