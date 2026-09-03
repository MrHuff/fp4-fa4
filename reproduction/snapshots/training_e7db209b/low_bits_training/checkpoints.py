#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Callable, Protocol
from dataclasses import dataclass
from pathlib import Path
import shutil
import os

import torch
import transformers

import torchao


from .models import TransformerModelArgs
from .config import JobConfig
from .utils import load_config
from .device_patch import get_device_info
from .ema_checkpoint import dcp_load_helper


from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.tools.logging import logger


from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters

from .utils import get_parallel_dims

_REPO_ROOT = Path(__file__).resolve().parents[1]

TensorDict = dict[str, torch.Tensor]

#  Register safe types needed to load FP8 and MXFP checkpoints
torch.serialization.add_safe_globals(
    [
        torchao.float8.fsdp_utils.WeightWithDynamicFloat8CastTensor,
    ]
)


@dataclass
class CheckpointLoader:
    """A class which helps to load and convert checkpoints"""

    model_name: str
    tt_to_transformers_config: (
        None | Callable[[JobConfig], transformers.PretrainedConfig]
    ) = None
    transformers_model_class: None | transformers.PreTrainedModel = None
    transformers_from_tt_checkpoint: (
        None | Callable[[transformers.PreTrainedModel, str], transformers.PreTrainedModel]
    ) = None
    tt_to_transformers_state_dict: None | Callable[[TensorDict], TensorDict] = None
    hf_reference_model: str | None = None
    tt_config_class = None
    tt_model_class = None

    def _check_assigned(self, attrs: list[str]):
        """
        Use this function to check that specific optional values have been defined
        before calling them
        """
        missing_attrs = []
        for optional_definition in attrs:
            if getattr(self, optional_definition) is None:
                missing_attrs.append(optional_definition)

        if missing_attrs:
            raise ValueError(
                f"{missing_attrs} was not defined for {self.model_name} update the converter initialisation"
            )

    def _get_device(self, job_config: JobConfig | str):
        """Placeholder for device selection based on the config currently targets CPU"""
        return "cpu"

    def load_transformers_model(self, job_config: JobConfig | str, checkpoint_path: str):
        self._check_assigned(
            [
                "tt_to_transformers_config",
                "transformers_model_class",
                "transformers_from_tt_checkpoint",
            ]
        )
        job_config = load_config(job_config, allow_upgrade=True)

        transformers_config = self.tt_to_transformers_config(job_config)
        with torch.device("meta"):
            transformers_model = self.transformers_model_class(transformers_config)
        device = self._get_device(job_config)
        # Find any temporary buffers - we need to check that they are correctly instantiated
        # after the model has been loaded.
        buffers_to_reinstantiate = []
        for name, buffer in transformers_model.named_buffers():
            logger.info(
                f"buffer: {name} is on {buffer.device} and has size {buffer.shape}. The buffer "
                "will need to be reinstanciated on the correct device."
            )
            buffers_to_reinstantiate.append(name)
        # Move the model to the correct device
        transformers_model = transformers_model.to_empty(device=device)
        buffer_values = {}
        for name, buffer in transformers_model.named_buffers():
            buffer_values[name] = buffer

        transformers_model = self.transformers_from_tt_checkpoint(
            transformers_model, checkpoint_path
        )
        # Check that the values of the buffers have changed - if they haven't then the checkpoint loading might
        # have not been done correctly.
        for name, buffer in transformers_model.named_buffers():
            if name in buffers_to_reinstantiate:
                if torch.allclose(buffer, buffer_values[name]):
                    logger.warning(
                        f"buffer: {name} has not been updated by the checkpoint loader "
                        f"- check the implementation of {self.transformers_from_tt_checkpoint}"
                    )

        return transformers_model

    def load_transformers_tokenizer(self):
        self._check_assigned(["hf_reference_model"])
        return transformers.AutoTokenizer.from_pretrained(self.hf_reference_model)

    def load_tt_model_and_tokenizer(
        self, job_config: JobConfig | str, checkpoint_path: str
    ) -> tuple[torch.nn.Module, Tokenizer]:
        job_config = load_config(job_config, allow_upgrade=True)
        train_spec = get_train_spec(job_config.model.name)
        model_config, tokenizer = tt_model_config_and_tokenizers_from_job_config(
            job_config
        )
        model_cls = train_spec.model_cls
        init_device = "meta"
        with torch.device(init_device):
            # logger.info(f"Init model on init_device: {init_device}")
            tt_model = model_cls(model_config)
        device = self._get_device(job_config)
        tt_model = tt_model.to_empty(device=device)
        dcp_load_helper(tt_model.state_dict(), checkpoint_id=checkpoint_path)
        return tt_model, tokenizer


def load_tt_tokenizer(job_config: JobConfig | str) -> Tokenizer:
    """Loads the tokenizer from a config file or config object"""
    # Load config from file if it is a string
    job_config = load_config(job_config, allow_upgrade=True)
    # load the tokenizer
    train_spec = get_train_spec(job_config.model.name)
    return train_spec.build_tokenizer_fn(job_config)


def load_tt_model_and_tokenizer(
    job_config: JobConfig | str,
    world_size: int | None = None,
    init_device: str = "cpu",
    buffer_device: str | None = None,
):
    """Loads the torchtitan model object based on the config

    Arguments:
        world_size: the number of accelerators
        init_device: The device on which to initialise the weights
        buffer_device: The device on which you want to execute the model (None means the same
            init_device).
    """
    job_config = load_config(job_config, allow_upgrade=True)

    if world_size is None:
        world_size = int(os.getenv("WORLD_SIZE", 1))
    parallel_dims = get_parallel_dims(job_config)
    device_type, device_module = get_device_info()
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"{device_type}:{int(os.getenv('LOCAL_RANK', 0))}")
    device_module.set_device(device)
    world_mesh = None
    distinct_seed_mesh_dims = []
    if world_size > 1:
        dist_utils.init_distributed(job_config)
        # build meshes
        world_mesh = parallel_dims.build_mesh(device_type=device_type)
        distinct_seed_mesh_dims = list(world_mesh.mesh_dim_names)

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    # NOTE: Currently setting distinct seeds over all mesh dimensions. We might only care about some.
    dist_utils.set_determinism(
        world_mesh, device, job_config.debug, distinct_seed_mesh_dims
    )
    train_spec = get_train_spec(job_config.model.name)

    # build dataloader
    tokenizer = train_spec.build_tokenizer_fn(job_config)

    # build model (using meta init)
    model_cls = train_spec.model_cls
    model_config = train_spec.model_args[job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.get_vocab_size()
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(
        f"Building {job_config.model.name} {job_config.model.flavor} with {model_config}"
    )
    with torch.device("meta"):
        model = model_cls(model_config)

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # apply parallelisms and initialization
    assert not parallel_dims.pp_enabled

    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    train_spec.parallelize_fn(model, parallel_dims, job_config)

    model.to_empty(device=init_device)
    with torch.no_grad():
        model.init_weights(buffer_device=buffer_device)
    model.eval()
    return model, tokenizer


def llama3_tt_to_transformers_config(
    model_config: TransformerModelArgs, tokenizer: Tokenizer
) -> transformers.LlamaConfig:
    """Convert a llama3 model config into the transformers config format"""

    # Torchtitan calculates the intermediate dimension using this sequence of operations
    # Check the model implementation for details.
    def torchtitan_intermediate_size(dim, ffn_multiplier, multiple_of):
        post_ffn = int(ffn_multiplier * int(2 * (4 * dim) / 3))
        return multiple_of * ((post_ffn + multiple_of - 1) // multiple_of)

    # These parameters are not set by our configuration - we set them explicitely from
    # Llama 3.1 8B: https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    llama_8b_upstream = dict(
        hidden_act="silu",
        max_position_embeddings=131072,
        mlp_bias=False,
        pretraining_tp=1,
        tie_word_embeddings=False,
        use_cache=True,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,  # this value in transformers is 0.02 - but also can be modified per layer with depth init.
    )
    transformers_config = transformers.LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.dim,
        intermediate_size=torchtitan_intermediate_size(
            model_config.dim, model_config.ffn_dim_multiplier, model_config.multiple_of
        ),
        num_hidden_layers=model_config.n_layers,
        num_attention_heads=model_config.n_heads,
        num_key_value_heads=model_config.n_kv_heads,
        rms_norm_eps=model_config.norm_eps,
        # no pad_id in tokenizer => eos_id by default.
        pad_token_id=tokenizer.eos_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        rope_theta=model_config.rope_theta,
        head_dim=model_config.dim // model_config.n_heads,
        rope_scaling={
            "rope_type": "default"
        },  # our implementation is closer to default than llama3
        **llama_8b_upstream,
    )
    return transformers_config


def deepseekv3_tt_to_transformers_config(
    model_config: TransformerModelArgs, tokenizer: Tokenizer
):
    raise NotImplementedError("Conversion does not exist yet")


class StateDictConverter(Protocol):
    """A class which helps to load and convert checkpoints"""

    @classmethod
    def transformers_from_tt_checkpoint(cls, transformers_model, checkpoint_path):
        """Loads a torchtitan distcp checkpoint into a transformers model"""
        ...


def rename_state_dict_entry(remap_keys: dict[str, str], state_dict: TensorDict):
    renamed_dict: TensorDict = {}
    for key, value in state_dict.items():
        new_key = key
        for sub, new_sub in remap_keys.items():
            if sub in key:
                new_key = new_key.replace(sub, new_sub)
        logger.debug("Key '%s' renamed to '%s'", key, new_key)
        renamed_dict[new_key] = value
    return renamed_dict


class Llama3StateDictConverter(StateDictConverter):
    """A class which helps to load and convert checkpoints for the llama3 model"""

    # In checkpoint
    #  'freq_cis'
    #  ...
    #  'layers.0.attention.wk.weight',
    #  'layers.0.attention.wo.weight',
    #  'layers.0.attention.wq.weight',
    #  'layers.0.attention.wv.weight',
    #  'layers.0.attention_norm.weight',
    #  'layers.0.feed_forward.w1.weight',
    #  'layers.0.feed_forward.w2.weight',
    #  'layers.0.feed_forward.w3.weight',
    #  'layers.0.ffn_norm.weight',
    #  ...
    #  'norm.weight',
    #  'output.weight',
    #  'tok_embeddings.weight'

    # In transformers:
    # 'lm_head.weight',
    #  'model.embed_tokens.weight',
    #  ...
    #  'model.layers.0.input_layernorm.weight',
    #  'model.layers.0.mlp.down_proj.weight',
    #  'model.layers.0.mlp.gate_proj.weight',
    #  'model.layers.0.mlp.up_proj.weight',
    #  'model.layers.0.post_attention_layernorm.weight',
    #  'model.layers.0.self_attn.k_proj.weight',
    #  'model.layers.0.self_attn.o_proj.weight',
    #  'model.layers.0.self_attn.q_proj.weight',
    #  'model.layers.0.self_attn.v_proj.weight',
    #  ...
    #  'model.norm.weight'
    remap_keys = {
        "embed_tokens": "tok_embeddings",
        "lm_head": "model.output",
        "self_attn.q_proj": "attention.wq",
        "self_attn.k_proj": "attention.wk",
        "self_attn.v_proj": "attention.wv",
        "self_attn.o_proj": "attention.wo",
        "post_attention_layernorm": "ffn_norm",
        "input_layernorm": "attention_norm",
        "mlp.gate_proj": "feed_forward.w1",
        "mlp.down_proj": "feed_forward.w2",
        "mlp.up_proj": "feed_forward.w3",
    }

    @classmethod
    def to_transformer_rotation_impl(cls, tt_weights: torch.Tensor, head_dim):
        """
        Re-order Q and V projections from TorchTitan to Transformers format

        The llama implementation in Transformers and Torchtian are different when it comes
        to applying rotary position encodings
        TorchTitan uses `apply_rotary_possition_embeddings` which reshapes the Q and Ks with:
        torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

        The re-ordering of Q and V projections is needed because the Llama stack (including
        Torchtitan) and transformers apply rotary position embeddings differently:

        - Torchtitan: xq.reshape(*xq.shape[:-1], -1, 2)
        - Transformers: torch.cat([-xq[..., :/2], xq[..., /2:]], dim=-1)

        The reshape places neighbouring elements on the same row,
        while the concatenation places elements in the second half of the head to the
        corresponding element in the first half.

        To fix this difference we re-order the rows of wq and wk

        Example:
        The first dimension is in the following orders for a head size of 6:
        TorchTitan:   1, 2, 3, 4, 5, 6 | 7, 8, 9 ...
        It needs to be reordered to:
        Transformers: 1, 3, 5, 2, 4, 6 | 7, 9, 11 ...

        """
        tt_weights = tt_weights.reshape((-1, head_dim, *tt_weights.shape[1:]))
        ind = torch.arange(tt_weights.shape[1] // 2, dtype=torch.long) * 2
        return torch.cat(
            (tt_weights[:, ind, ...], tt_weights[:, ind + 1, ...]), dim=1
        ).flatten(0, 1)

    @classmethod
    def transformers_from_tt_checkpoint(cls, transformers_model, checkpoint_path):
        """Loads a torchtitan distcp checkpoint into a transformers model

        This requires 2 conversions:

        1. Rename all the tensors - this is a straight-forward mapping
        2. Reorder the Q and V projection weights to match the transformers implementation.

        """
        # We need to re-instantiate the rotary embeddings as the inv_freq buffer does not
        # support being instantiated on the meta device.
        transformers_model.model.rotary_emb = type(transformers_model.model.rotary_emb)(
            transformers_model.model.rotary_emb.config,
            device=transformers_model.model.device,
        )
        # DCP load expects the checkpoint and the statedict to be in the same format
        # To do that we need the keys of the state dict to match those of the checkpoint.
        dcp_load_helper(
            rename_state_dict_entry(cls.remap_keys, transformers_model.state_dict()),
            checkpoint_id=checkpoint_path,
        )

        reorder_tensors_func = {
            "self_attn.q_proj": cls.to_transformer_rotation_impl,
            "self_attn.k_proj": cls.to_transformer_rotation_impl,
        }

        for name, param in transformers_model.named_parameters():
            for stub, reorder_func in reorder_tensors_func.items():
                if stub in name:
                    param.data = reorder_func(
                        param.data, transformers_model.config.head_dim
                    )
                    logger.debug("Reordered parameter: %s", name)
                    break

        return transformers_model


class DeepSeekV3StateDictConverter(StateDictConverter):
    @classmethod
    def transformers_from_tt_checkpoint(cls, transformers_model, checkpoint_path):
        raise NotImplementedError("Conversion not implemented")


def tt_model_config_and_tokenizers_from_job_config(
    job_config: JobConfig,
) -> tuple[TransformerModelArgs, Tokenizer]:
    """This function replicates some of the logic in train.py to allow us
    to create the model config."""
    train_spec = get_train_spec(job_config.model.name)
    _make_tokenizer_path_absolute(job_config)
    tokenizer = train_spec.build_tokenizer_fn(job_config)
    model_config = train_spec.model_args[job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.get_vocab_size()
    model_config.max_seq_len = job_config.training.seq_len
    return model_config, tokenizer


def _make_tokenizer_path_absolute(job_config: JobConfig):
    """Makes old and new style tokenizers absolute path - useful when running from outside
    the repository root folder."""

    def try_absolute(path):
        tokenizer_path = Path(path)
        if not tokenizer_path.exists():
            tokenizer_path = _REPO_ROOT / path
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer could not be found at {path} or {tokenizer_path}"
                )
        return str(tokenizer_path)

    if job_config.model.tokenizer_path:  # old format tokenizer
        job_config.model.tokenizer_path = try_absolute(job_config.model.tokenizer_path)
    else:
        job_config.model.hf_assets_path = try_absolute(job_config.model.hf_assets_path)


def convert_checkpoint_to_transformers(
    checkpoint_path: str,
    job_config: JobConfig | str,
    converted_checkpoint_dir: str | None = None,
    overwrite: bool = False,
):
    """Convert a checkpoint to the transformers format"""
    if converted_checkpoint_dir is None:
        converted_checkpoint_dir = "transformers-checkpoint"
    target_dir = Path(checkpoint_path) / converted_checkpoint_dir
    # Make sure a bad model checkpoint gets deleted (can happen if pre-empted during conversion)
    incomplete_checkpoint = target_dir / ".incomplete_checkpoint"
    valid_checkpoint_exists = (
        target_dir.exists()
        and list(target_dir.glob("*safetensors"))
        and (not incomplete_checkpoint.exists())
    )
    if (not overwrite) and valid_checkpoint_exists:
        logger.info(f"Converted checkpoint directory {target_dir} already exists")
        return target_dir
    # Create the directory and file if it doesn't exist
    incomplete_checkpoint.parent.mkdir(exist_ok=True, parents=True)
    incomplete_checkpoint.touch()
    # clean the target dir of partial checkpoints to avoid clashes
    for file_to_clean in [
        target_dir / t
        for t in [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.model",
        ]
    ] + list(target_dir.glob("*safetensors")):
        if file_to_clean.exists():
            file_to_clean.unlink()
    job_config = load_config(job_config, allow_upgrade=True, validate=False)
    _make_tokenizer_path_absolute(job_config)

    logger.info(f"Converting checkpoint to transformers format at {target_dir}")
    checkpoint_loader = REGISTERED_CHECKPOINT_CONVERSIONS[job_config.model.name]
    transformers_model = checkpoint_loader.load_transformers_model(
        job_config, checkpoint_path
    )
    transformers_model.save_pretrained(target_dir)
    # Transformers can load tiktoken tokenizers from the tokenizer.model file
    # so we need to copy that file to the target directory
    if job_config.model.hf_assets_path:
        tokenizer_hf_path = Path(job_config.model.hf_assets_path)
        for file in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            shutil.copy(tokenizer_hf_path / file, target_dir / file)
    elif job_config.model.tokenizer_path:  # legacy tokenizer
        shutil.copy(job_config.model.tokenizer_path, target_dir / "tokenizer.model")
    (target_dir / ".incomplete_checkpoint").unlink()
    return target_dir


REGISTERED_CHECKPOINT_CONVERSIONS = {
    "llama3_gc": CheckpointLoader(
        model_name="llama3_gc",
        tt_to_transformers_config=lambda job_config: llama3_tt_to_transformers_config(
            *tt_model_config_and_tokenizers_from_job_config(job_config)
        ),
        transformers_model_class=transformers.LlamaForCausalLM,
        transformers_from_tt_checkpoint=Llama3StateDictConverter.transformers_from_tt_checkpoint,
        hf_reference_model="meta-llama/Llama-3.1-8B",
    )
    # "deepseek_v3_gc": CheckpointLoader(
    #     model_name="deepseek_v3_gc",
    #     tt_to_transformers_config=lambda job_config: deepseekv3_tt_to_transformers_config(
    #         *tt_model_config_and_tokenizers_from_job_config(job_config)
    #     ),
    #     transformers_model_class=transformers.DeepseekV3ForCausalLM,
    #     transformers_from_tt_checkpoint=Llama3StateDictConverter.transformers_from_tt_checkpoint,
    #     hf_reference_model="deepseek-ai/DeepSeek-V3",
    # ),
}
