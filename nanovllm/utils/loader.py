import os
from dataclasses import dataclass, field
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))


@dataclass
class WeightLoadReport:
    loaded_keys: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    tied_aliases: list[str] = field(default_factory=list)


def load_qwen3_vl_weights(
    model: nn.Module, path: str, *, text_only: bool = False
) -> WeightLoadReport:
    """Audit all headers, then stream into the TP=1 VL parameter namespace.

    Header errors are rejected before any parameter is written. A runtime
    copy/device error can leave a partial load; the caller must discard it.
    """
    sources, targets, report = _audit_qwen3_vl_weights(model, path, text_only=text_only)
    for target in targets.values():
        if target.parameter.is_meta:
            raise ValueError(
                f"cannot load into meta parameter {target.name}; materialize the model first"
            )
    for file in sorted(set(sources.values())):
        with safe_open(file, "pt", "cpu") as reader:
            for name in sorted(sources):
                if sources[name] != file:
                    continue
                target = targets[name]
                weight = reader.get_tensor(name)
                try:
                    loader = getattr(
                        target.parameter, "weight_loader", default_weight_loader
                    )
                    if target.shard_id is None:
                        loader(target.parameter, weight)
                    else:
                        loader(target.parameter, weight, target.shard_id)
                except Exception as error:
                    raise ValueError(
                        f"shard load failed: {name} -> {target.name}: {error}"
                    ) from error
                finally:
                    del weight
                report.loaded_keys.append(name)
    return report


@dataclass
class _WeightTarget:
    name: str
    parameter: nn.Parameter
    shard_id: str | int | None
    shape: tuple[int, ...]


def _qwen3_vl_targets(model: nn.Module, text_only: bool):
    from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear

    parameters = dict(model.named_parameters(remove_duplicate=False))
    embedding = "model.language_model.embed_tokens.weight"
    head = "lm_head.weight"
    aliases = []
    if head in parameters:
        if embedding not in parameters or parameters[head] is not parameters[embedding]:
            raise ValueError(
                f"tied {head} and {embedding} must share the same Parameter"
            )
        aliases.append(head)
    targets, seen = {}, {}

    def add_target(source, target):
        if source in targets:
            raise ValueError(
                f"duplicate model mapping for {source}: {targets[source].name} and {target.name}"
            )
        targets[source] = target

    for name, parameter in parameters.items():
        if name == head or (text_only and name.startswith("model.visual.")):
            continue
        if not name.startswith(("model.language_model.", "model.visual.")):
            raise ValueError(f"unexpected model parameter namespace: {name}")
        if id(parameter) in seen:
            raise ValueError(
                f"duplicate parameter alias: {name} and {seen[id(parameter)]}"
            )
        seen[id(parameter)] = name
        module_name, suffix = name.rsplit(".", 1)
        owner = model.get_submodule(module_name)
        if getattr(owner, "tp_size", 1) != 1:
            raise ValueError(f"unsupported TP shard for {name}; requires TP=1")
        packed = None
        if name.startswith("model.language_model."):
            if module_name.endswith(".self_attn.qkv_proj"):
                if not isinstance(owner, QKVParallelLinear):
                    raise ValueError(f"unexpected packed shard module: {module_name}")
                packed = [
                    ("q_proj", "q", owner.num_heads * owner.head_size),
                    ("k_proj", "k", owner.num_kv_heads * owner.head_size),
                    ("v_proj", "v", owner.num_kv_heads * owner.head_size),
                ]
            elif module_name.endswith(".mlp.gate_up_proj"):
                if (
                    not isinstance(owner, MergedColumnParallelLinear)
                    or len(owner.output_sizes) != 2
                ):
                    raise ValueError(f"unexpected packed shard module: {module_name}")
                packed = [
                    ("gate_proj", 0, owner.output_sizes[0]),
                    ("up_proj", 1, owner.output_sizes[1]),
                ]
        if packed is None:
            add_target(
                name, _WeightTarget(name, parameter, None, tuple(parameter.shape))
            )
            continue
        if not callable(getattr(parameter, "weight_loader", None)):
            raise ValueError(f"missing packed shard weight_loader: {name}")
        if parameter.ndim not in (1, 2) or parameter.shape[0] != sum(
            size for _, _, size in packed
        ):
            raise ValueError(
                f"shape mismatch for packed shard target {name}: {tuple(parameter.shape)}"
            )
        parent = module_name.rsplit(".", 1)[0]
        for projection, shard, size in packed:
            source = f"{parent}.{projection}.{suffix}"
            add_target(
                source,
                _WeightTarget(name, parameter, shard, (size, *parameter.shape[1:])),
            )
    return targets, aliases


def _audit_qwen3_vl_weights(model: nn.Module, path: str, *, text_only: bool = False):
    """Metadata-only preflight, also usable on a test-only meta schema."""
    targets, aliases = _qwen3_vl_targets(model, text_only)
    report = WeightLoadReport(tied_aliases=aliases)
    sources, seen, errors = {}, set(), []
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"missing safetensors shard files: {path}")
    for file in files:
        with safe_open(file, "pt", "cpu") as reader:
            for name in reader.keys():
                if name in seen:
                    errors.append(f"duplicate checkpoint shard: {name} in {file}")
                    continue
                seen.add(name)
                if text_only and name.startswith("model.visual."):
                    report.skipped_keys.append(name)
                    continue
                if name not in targets:
                    errors.append(f"unexpected checkpoint key: {name}")
                    continue
                shape = tuple(reader.get_slice(name).get_shape())
                if shape != targets[name].shape:
                    errors.append(
                        f"shape mismatch: {name} has {shape}, expected {targets[name].shape}"
                    )
                sources[name] = file
    missing = sorted(set(targets) - set(sources))
    if missing:
        errors.append(f"missing checkpoint shards: {', '.join(missing)}")
    if errors:
        raise ValueError("; ".join(errors))
    report.skipped_keys.sort()
    return sources, targets, report
