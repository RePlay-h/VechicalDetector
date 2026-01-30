from pathlib import Path
import sys
import subprocess

def test_train_smoke_create_artifact(tmp_path: Path) -> None:
    outdir = tmp_path / "models" / "smoke"
    params_path = tmp_path / "params.yaml"

    params_path.write_text(
        "\n".join(
            [
                "train:",
                "  experiment_name: test_exp",
                "  run_name: test_run",
                "  metric_value: 0.5",
                "  artifact_text: test",
                f"  model_path: {outdir}",
            ]
        ),
    )

    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "ml/src/smoke_train.py",
        "--params",
        str(params_path),
    ]

    subprocess.check_call(cmd)

    assert (outdir / "SMOKE_ARTIFACT.txt").exists(), "Smoke artifact file was not created"
