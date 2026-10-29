import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Demos directory (e.g., "ricl_droid_preprocessing/collected_demos/2025-03-04").
    demos_dir: str
    # Retrieval mode for LIBERO RICL serving: "nearest" or "random".
    retrieval_mode: str = "nearest"
    # Seed used when retrieval_mode="random".
    retrieval_seed: int = 0
    # Retrieval representation used by the LIBERO comparison experiment.
    retrieval_method: str = "dino"
    # "task" searches only the query task; "global" searches every task in demos_dir.
    retrieval_scope: str = "task"
    # Completed candidate feature cache for Pi first-fusion retrieval.
    pi_feature_dir: str | None = None
    # Normal Pi0 encoder identity used by the completed offline benchmark.
    pi_retriever_config: str = "pi0_fast_libero90_normal"
    pi_retriever_checkpoint: str | None = None
    pi_retriever_dtype: str = "float32"
    # Checkpoint providing state normalization for DINO+state.
    state_norm_checkpoint: str | None = None
    dino_state_beta: float = 1.0
    # Candidate distance backend and bounded transfer chunk size.
    retrieval_search_device: str = "auto"
    retrieval_distance_chunk_size: int = 256


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.DROID

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_ricl_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        demos_dir=args.policy.demos_dir,
        retrieval_mode=args.policy.retrieval_mode,
        retrieval_seed=args.policy.retrieval_seed,
        retrieval_method=args.policy.retrieval_method,
        retrieval_scope=args.policy.retrieval_scope,
        pi_feature_dir=args.policy.pi_feature_dir,
        pi_retriever_config=args.policy.pi_retriever_config,
        pi_retriever_checkpoint=args.policy.pi_retriever_checkpoint,
        pi_retriever_dtype=args.policy.pi_retriever_dtype,
        state_norm_checkpoint=args.policy.state_norm_checkpoint,
        dino_state_beta=args.policy.dino_state_beta,
        retrieval_search_device=args.policy.retrieval_search_device,
        retrieval_distance_chunk_size=args.policy.retrieval_distance_chunk_size,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
