"""Pokemon Deep CFR 학습 — ClearML + RTX 5090.

Usage:
    python train_pokecfr.py
"""

import torch
from clearml import Task

Task.add_requirements("torch")
Task.add_requirements("maturin")
task = Task.init(project_name="PokeCFR", task_name="Deep CFR v1 — Gen9 OU")

# 하이퍼파라미터를 ClearML에 등록 (UI에서 수정 가능)
params = {
    # Encoding
    "species_dim": 32,
    "type_dim": 8,
    "move_dim": 16,
    "item_dim": 16,
    "d_model": 128,
    "num_heads": 4,
    "num_attn_layers": 2,
    # Network
    "hidden_dim": 256,
    "num_layers": 3,
    "num_actions": 9,
    # Training
    "num_cfr_iterations": 200,
    "num_traversals_per_iter": 100,
    "advantage_buffer_size": 2_000_000,
    "strategy_buffer_size": 2_000_000,
    "max_depth": 40,
    "learning_rate": 1e-3,
    "batch_size": 2048,
    "num_epochs_per_iter": 2,
    "seed": 42,
}
task.connect(params)

task.execute_remotely(queue_name="junha-5090")

# === 아래부터 RTX 5090에서 실행 ===

# poke-engine 빌드 (서버에 없으면 자동 빌드)
import subprocess, os, sys
try:
    import poke_engine
except ImportError:
    print("poke-engine not found, building from source...")
    # Rust 설치 확인
    if subprocess.run(["which", "cargo"], capture_output=True).returncode != 0:
        print("Installing Rust...")
        subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                       shell=True, check=True)
        os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]

    # poke-engine 클론 및 빌드
    if os.path.exists("poke-engine"):
        import shutil
        shutil.rmtree("poke-engine")
    subprocess.run(["git", "clone", "https://github.com/pmariglia/poke-engine.git"], check=True)

    # Cargo.toml을 Gen9로 수정
    cargo_toml = "poke-engine/poke-engine-py/Cargo.toml"
    with open(cargo_toml, "r") as f:
        content = f.read()
    content = content.replace('default = ["poke-engine/gen4"]',
                              'default = ["poke-engine/gen9", "poke-engine/terastallization"]')
    with open(cargo_toml, "w") as f:
        f.write(content)

    # 커스텀 바인딩 패치 적용
    subprocess.run([sys.executable, "patch_poke_engine.py"], check=True)

    # 빌드 (maturin을 python -m 으로 실행)
    subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
    subprocess.run([sys.executable, "-m", "maturin", "develop", "--release"],
                   cwd="poke-engine/poke-engine-py", check=True)
    print("poke-engine built successfully!")

import random
from poke_engine import State, Side, Pokemon, Move

from pokecfr.config import EncodingConfig, NetworkConfig, TrainingConfig
from pokecfr.trainer import run_deep_cfr

# ClearML 파라미터 → config 객체
enc_cfg = EncodingConfig(
    species_dim=params["species_dim"],
    type_dim=params["type_dim"],
    move_dim=params["move_dim"],
    item_dim=params["item_dim"],
    d_model=params["d_model"],
    num_heads=params["num_heads"],
    num_attn_layers=params["num_attn_layers"],
)
net_cfg = NetworkConfig(
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"],
    num_actions=params["num_actions"],
)
train_cfg = TrainingConfig(
    num_cfr_iterations=params["num_cfr_iterations"],
    num_traversals_per_iter=params["num_traversals_per_iter"],
    advantage_buffer_size=params["advantage_buffer_size"],
    strategy_buffer_size=params["strategy_buffer_size"],
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    batch_size=params["batch_size"],
    num_epochs_per_iter=params["num_epochs_per_iter"],
    seed=params["seed"],
)

torch.manual_seed(train_cfg.seed)
random.seed(train_cfg.seed)


# ── OU 팀 풀 (학습용 샘플 팀들) ──
def make_pokemon(name, types, hp, atk, defe, spa, spd, spe, moves, ability="none", item="none"):
    ms = [Move(id=m, pp=16) for m in moves]
    while len(ms) < 4:
        ms.append(Move(id="none", pp=0))
    return Pokemon(
        id=name, level=100, types=types, hp=hp, maxhp=hp,
        attack=atk, defense=defe, special_attack=spa,
        special_defense=spd, speed=spe, ability=ability,
        item=item, nature="serious", evs=(0, 0, 0, 0, 0, 0),
        moves=ms, status="none",
    )


# 대표적인 OU 포켓몬 풀
OU_POOL = [
    make_pokemon("garchomp", ("ground", "dragon"), 357, 359, 226, 176, 206, 333,
                 ["earthquake", "outrage", "swordsdance", "stoneedge"], "roughskin", "focussash"),
    make_pokemon("dragonite", ("dragon", "flying"), 386, 367, 226, 236, 236, 196,
                 ["extremespeed", "earthquake", "dragondance", "roost"], "multiscale", "heavydutyboots"),
    make_pokemon("heatran", ("fire", "steel"), 386, 176, 253, 359, 253, 183,
                 ["magmastorm", "earthpower", "stealthrock", "flashcannon"], "flashfire", "leftovers"),
    make_pokemon("ferrothorn", ("grass", "steel"), 352, 214, 329, 144, 268, 40,
                 ["stealthrock", "leechseed", "gyroball", "knockoff"], "ironbarbs", "leftovers"),
    make_pokemon("rotomwash", ("electric", "water"), 304, 122, 253, 246, 253, 192,
                 ["hydropump", "voltswitch", "willowisp", "thunderbolt"], "levitate", "leftovers"),
    make_pokemon("scizor", ("bug", "steel"), 344, 359, 236, 131, 196, 166,
                 ["bulletpunch", "uturn", "swordsdance", "knockoff"], "technician", "choiceband"),
    make_pokemon("tyranitar", ("rock", "dark"), 404, 367, 256, 226, 236, 158,
                 ["stoneedge", "crunch", "earthquake", "stealthrock"], "sandstream", "leftovers"),
    make_pokemon("clefable", ("fairy", "typeless"), 394, 158, 240, 226, 216, 156,
                 ["moonblast", "softboiled", "calmmind", "thunderwave"], "magicguard", "leftovers"),
    make_pokemon("toxapex", ("poison", "water"), 304, 152, 307, 132, 322, 70,
                 ["scald", "recover", "toxicspikes", "haze"], "regenerator", "blacksludge"),
    make_pokemon("landorustherian", ("ground", "flying"), 354, 341, 220, 226, 196, 262,
                 ["earthquake", "uturn", "stealthrock", "stoneedge"], "intimidate", "choicescarf"),
    make_pokemon("slowking", ("water", "psychic"), 394, 167, 196, 236, 256, 96,
                 ["scald", "psychic", "slackoff", "thunderwave"], "regenerator", "heavydutyboots"),
    make_pokemon("weavile", ("dark", "ice"), 344, 372, 166, 101, 206, 349,
                 ["knockoff", "iceshard", "swordsdance", "tripleaxel"], "pressure", "choiceband"),
]


def make_random_team() -> list:
    """OU 풀에서 랜덤 6마리 선택."""
    return random.sample(OU_POOL, min(6, len(OU_POOL)))


def initial_state_fn() -> State:
    """새 게임 상태 생성 (랜덤 팀 매칭)."""
    return State(
        side_one=Side(pokemon=make_random_team()),
        side_two=Side(pokemon=make_random_team()),
    )


# ── 학습 실행 ──
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"CFR iterations: {train_cfg.num_cfr_iterations}")
print(f"Traversals/iter: {train_cfg.num_traversals_per_iter}")
print(f"Max depth: {train_cfg.max_depth}")
print(f"Encoding: pokemon_dim={enc_cfg.pokemon_dim}, infoset_dim={enc_cfg.infoset_dim}")
print()

strategy_net = run_deep_cfr(
    initial_state_fn=initial_state_fn,
    enc_cfg=enc_cfg,
    net_cfg=net_cfg,
    train_cfg=train_cfg,
    logger=task.get_logger(),
)

# 최종 모델 아티팩트 업로드
task.upload_artifact("strategy_net", artifact_object="strategy_net_final.pt")
for ckpt in __import__("glob").glob("checkpoint_iter_*.pt"):
    task.upload_artifact(f"checkpoint_{ckpt}", artifact_object=ckpt)

print("학습 완료!")
