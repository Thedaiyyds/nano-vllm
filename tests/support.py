"""Small validation workers; never load a model during test collection."""

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlparse

LEGACY_IDS = [9707, 11, 50493, 8273, 4086, 44, 13]
LEGACY_SAMPLING = dict(temperature=0.7, max_tokens=8, ignore_eos=False)
LEGACY_SEED = 2026
LEGACY_REFERENCE_REVISION = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
LEGACY_OPTIONS = dict(
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=64,
    max_num_batched_tokens=64,
    max_num_seqs=1,
    gpu_memory_utilization=0.2,
)


def run_worker(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        check=False,
        timeout=120,
        capture_output=True,
        text=True,
    )


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_legacy_record(record: dict) -> None:
    """Reject incomplete or malformed output before it can become a baseline."""
    assert record["prompt_ids"] == LEGACY_IDS
    assert record["seed"] == LEGACY_SEED
    assert record["options"] == LEGACY_OPTIONS
    assert record["sampling"] == LEGACY_SAMPLING
    assert set(record["model_files_sha256"]) == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert isinstance(record["text"], str)
    ids = record["token_ids"]
    assert isinstance(ids, list) and 0 < len(ids) <= 8
    assert all(
        type(token) is int and 0 <= token < record["vocab_size"] for token in ids
    )
    assert record["environment"] and record["model_sha256"] and record["source_root"]


def assert_legacy_match(
    actual: dict, expected: dict, *, require_reference: bool = False
) -> None:
    validate_legacy_record(actual)
    validate_legacy_record(expected)
    if require_reference:
        assert expected.get("source_revision") == LEGACY_REFERENCE_REVISION, (
            "original source revision mismatch"
        )
    for key in (
        "prompt_ids",
        "seed",
        "options",
        "environment",
        "model_sha256",
        "model_files_sha256",
        "sampling",
        "vocab_size",
    ):
        assert actual[key] == expected[key], (
            f"BLOCKED: baseline metadata mismatch: {key}"
        )
    assert actual["token_ids"] == expected["token_ids"], (
        "legacy fixed-input token regression"
    )


def legacy_worker(model: Path, output: Path, source_root: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "BLOCKED: real CUDA is unavailable; no model generation performed"
        )
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (model / name).is_file():
            raise FileNotFoundError(
                f"BLOCKED: local model file missing: {model / name}"
            )
    # The worker starts fresh, so this selects either original HEAD or current source.
    sys.path.insert(0, str(source_root.resolve()))
    import nanovllm
    from nanovllm import LLM, SamplingParams

    assert Path(nanovllm.__file__).resolve().is_relative_to(source_root.resolve())
    versions = {}
    for name in ("torch", "transformers", "flash-attn", "triton", "numpy"):
        dist = importlib.metadata.distribution(name)
        versions[name] = {
            "version": dist.version,
            "source": dist.read_text("direct_url.json"),
        }
    hf_source = json.loads(versions["transformers"]["source"] or "{}")
    if hf_source.get("dir_info", {}).get("editable"):
        hf_root = Path(unquote(urlparse(hf_source["url"]).path))
        versions["transformers"]["code_revision"] = subprocess.check_output(
            ["git", "-C", str(hf_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(hf_root), "diff", "HEAD", "--", "src/transformers"]
        )
        versions["transformers"]["code_dirty_sha256"] = hashlib.sha256(
            dirty
        ).hexdigest()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    environment = {
        "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "dtype": "bfloat16",
        "compile": False,
        "cuda_graph": False,
    }
    llm = LLM(str(model), **LEGACY_OPTIONS)
    try:
        assert next(llm.model_runner.model.parameters()).dtype == torch.bfloat16
        llm.model_runner.model.eval()
        torch.manual_seed(LEGACY_SEED)
        torch.cuda.manual_seed_all(LEGACY_SEED)
        result = llm.generate(
            [LEGACY_IDS], SamplingParams(**LEGACY_SAMPLING), use_tqdm=False
        )[0]
        torch.cuda.synchronize()
        record = dict(
            prompt_ids=LEGACY_IDS,
            seed=LEGACY_SEED,
            options=LEGACY_OPTIONS,
            sampling=LEGACY_SAMPLING,
            model_files_sha256={
                name: digest_file(model / name)
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
            },
            token_ids=result["token_ids"],
            text=result["text"],
            vocab_size=llm.model_runner.config.hf_config.vocab_size,
            environment=environment,
            model_sha256=digest_file(model / "model.safetensors"),
            source_root=str(source_root.resolve()),
        )
        validate_legacy_record(record)
    finally:
        # Original exit is not idempotent: explicit cleanup must unregister atexit.
        atexit.unregister(llm.exit)
        llm.exit()
    output.write_text(json.dumps(record, indent=2) + "\n")


def capture_legacy_baseline(model: Path, target: Path) -> None:
    """Explicit capture only: two fresh original-HEAD runs must match first."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED: real CUDA unavailable; baseline NOT captured")
    if target.exists():
        raise FileExistsError(
            "Refusing to overwrite a legacy baseline; review changes explicitly"
        )
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", LEGACY_REFERENCE_REVISION], cwd=root, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="qwen-legacy-original-") as temp:
        temp = Path(temp)
        archive = temp / "original.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", revision, "nanovllm"],
                cwd=root,
                stdout=stream,
                check=True,
            )
        with tarfile.open(archive) as stream:
            stream.extractall(temp, filter="data")
        records = []
        for index in range(2):
            out = temp / f"run-{index}.json"
            result = run_worker(
                "tests.support",
                ["--legacy-worker", str(model), str(out), "--source-root", str(temp)],
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            records.append(json.loads(out.read_text()))
        assert_legacy_match(records[0], records[1])
        records[0]["source_revision"] = revision
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records[0], indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--legacy-worker", nargs=2, metavar=("MODEL", "OUTPUT"))
    mode.add_argument("--capture-legacy-baseline", nargs=2, metavar=("MODEL", "TARGET"))
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    if args.legacy_worker:
        legacy_worker(*map(Path, args.legacy_worker), args.source_root)
    else:
        capture_legacy_baseline(*map(Path, args.capture_legacy_baseline))


if __name__ == "__main__":
    main()
