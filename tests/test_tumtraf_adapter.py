"""Smoke test for TUMTraf-V2X_mini"""
from __future__ import annotations

import argparse
from project_paths import dataset_path
import numpy as np

from adapters.tumtraf_adapter import TUMTrafAdapter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--config")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)

    adapter = TUMTrafAdapter(dataset_path("tumtraf", args.root, args.config), split=args.split)
    print(f"Dataset: {adapter}")
    # __getitem__ runs CanonicalV2XSample.validate() automatically.
    sample = adapter[args.index]
    print("VALIDATION: PASSED")
    print("Frame:", sample.source.frame_id)
    print("Timestamp:", sample.timestamp_s)
    print("Reference agent:", sample.reference_agent_id)
    print("Agents:", len(sample.agents))
    print("3D boxes:", sample.annotations.boxes_3d.shape)
    print("Native classes:", sorted(set(sample.annotations.native_labels)))
    for agent in sample.agents:
        print(f"\n{agent.agent_id} ({agent.agent_type})")
        print("  LiDAR XYZ:", agent.lidar.xyz.shape if agent.lidar else None)
        print("  Images with verified calibration:", len(agent.cameras))
        print("  Transform to reference:\n", agent.T_ref_from_agent)
        assert agent.lidar is not None and agent.lidar.xyz.shape[1] == 3
        assert np.isfinite(agent.lidar.xyz).all(), "Nonfinite LiDAR positions"
    assert sample.annotations.boxes_3d.shape[1] == 7
    assert np.isfinite(sample.annotations.boxes_3d).all(), "Nonfinite boxes"
    reference = next(agent for agent in sample.agents if agent.is_reference)
    assert np.allclose(reference.T_ref_from_agent, np.eye(4))
    print("\nSMOKE TEST: PASSED")
    print("Camera image paths available:", sum(
        p is not None for p in sample.extras["camera_image_paths_uncalibrated"].values()))


def test_validation(cli_args):
    main(cli_args("tumtraf"))


if __name__ == "__main__":
    main()
