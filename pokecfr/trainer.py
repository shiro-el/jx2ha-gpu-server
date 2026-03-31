"""Deep CFR 학습 루프.

각 CFR iteration:
1. Self-play traversal → advantage/strategy 버퍼에 샘플 수집
2. Advantage network 재학습 (전체 버퍼, iteration 가중 MSE)
3. 마지막에 strategy network 학습 (average strategy)
"""

import torch
import torch.optim as optim

from poke_engine import State as PEState

from pokecfr.buffers import AdvantageBuffer, StrategyBuffer
from pokecfr.config import EncodingConfig, NetworkConfig, TrainingConfig
from pokecfr.converter import state_to_infoset
from pokecfr.loss import advantage_loss, strategy_loss
from pokecfr.networks import AdvantageNetwork, StrategyNetwork, ValueNetwork
from pokecfr.traversal import traverse, _add_batch_dim


def collate_infosets(infosets: list[dict]) -> dict:
    """infoset dict 리스트 → 배치 텐서 dict."""
    result = {}
    for key in infosets[0]:
        if isinstance(infosets[0][key], dict):
            result[key] = collate_infosets([inf[key] for inf in infosets])
        elif isinstance(infosets[0][key], torch.Tensor):
            result[key] = torch.stack([inf[key] for inf in infosets])
        else:
            result[key] = infosets[0][key]
    return result


def train_advantage_net(
    net: AdvantageNetwork,
    buffer: AdvantageBuffer,
    cfg: TrainingConfig,
    device: torch.device,
    logger=None,
    iteration: int = 0,
) -> float:
    """Advantage network 학습. 전체 버퍼에서 epoch만큼 반복."""
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
    total_loss = 0.0
    num_batches = 0

    for _ in range(cfg.num_epochs_per_iter):
        batch = buffer.sample_batch(cfg.batch_size)
        if len(batch) == 0:
            continue

        # state 역직렬화 → infoset 텐서 변환
        infosets = []
        regrets = []
        masks = []
        iterations = []

        for sample in batch:
            state = PEState.from_string(sample["state_str"])
            infoset = state_to_infoset(state, sample["player"], sample["turn"])
            infosets.append(infoset)
            regrets.append(sample["regret"])
            masks.append(sample["action_mask"])
            iterations.append(sample["iteration"])

        batched_infoset = collate_infosets(infosets)
        batched_infoset = _to_device(batched_infoset, device)

        target_regret = torch.stack(regrets).to(device)
        action_mask = torch.stack(masks).to(device)
        iter_weights = torch.tensor(iterations, dtype=torch.float32, device=device)

        # Forward
        predicted = net(batched_infoset)

        # Loss
        loss = advantage_loss(predicted, target_regret, iter_weights, action_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if logger:
        logger.report_scalar("loss", "advantage", iteration=iteration, value=avg_loss)

    net.eval()
    return avg_loss


def train_strategy_net(
    net: StrategyNetwork,
    buffer: StrategyBuffer,
    cfg: TrainingConfig,
    device: torch.device,
    logger=None,
    num_epochs: int = 10,
) -> float:
    """Strategy network 학습. 전체 CFR 종료 후 한번."""
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
    total_loss = 0.0
    num_batches = 0

    for epoch in range(num_epochs):
        batch = buffer.sample_batch(cfg.batch_size)
        if len(batch) == 0:
            continue

        infosets = []
        strategies = []
        masks = []
        iterations = []
        reach_probs = []

        for sample in batch:
            state = PEState.from_string(sample["state_str"])
            infoset = state_to_infoset(state, sample["player"], sample["turn"])
            infosets.append(infoset)
            strategies.append(sample["strategy"])
            masks.append(sample["action_mask"])
            iterations.append(sample["iteration"])
            reach_probs.append(sample["reach_prob"])

        batched_infoset = collate_infosets(infosets)
        batched_infoset = _to_device(batched_infoset, device)

        target_strat = torch.stack(strategies).to(device)
        action_mask = torch.stack(masks).to(device)
        iter_weights = torch.tensor(iterations, dtype=torch.float32, device=device)
        rp = torch.tensor(reach_probs, dtype=torch.float32, device=device)

        predicted = net(batched_infoset, action_mask)

        loss = strategy_loss(predicted, target_strat, iter_weights, rp, action_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if logger:
            logger.report_scalar("loss", "strategy", iteration=epoch, value=loss.item())

    net.eval()
    return total_loss / max(num_batches, 1)


def _to_device(data: dict, device: torch.device) -> dict:
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _to_device(v, device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def run_deep_cfr(
    initial_state_fn,
    enc_cfg: EncodingConfig,
    net_cfg: NetworkConfig,
    train_cfg: TrainingConfig,
    logger=None,
):
    """Deep CFR 메인 학습 루프.

    Args:
        initial_state_fn: () → PEState, 새 게임 상태를 생성하는 함수
        enc_cfg, net_cfg, train_cfg: 설정
        logger: ClearML Logger (optional)

    Returns:
        학습된 StrategyNetwork
    """
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # 네트워크 초기화
    adv_nets = [
        AdvantageNetwork(enc_cfg, net_cfg).to(device),
        AdvantageNetwork(enc_cfg, net_cfg).to(device),
    ]
    value_net = ValueNetwork(enc_cfg, net_cfg).to(device)
    strategy_net = StrategyNetwork(enc_cfg, net_cfg).to(device)

    adv_buffers = [
        AdvantageBuffer(train_cfg.advantage_buffer_size),
        AdvantageBuffer(train_cfg.advantage_buffer_size),
    ]
    strat_buffer = StrategyBuffer(train_cfg.strategy_buffer_size)

    for t in range(1, train_cfg.num_cfr_iterations + 1):
        # ── Traversal phase ──
        for net in adv_nets:
            net.eval()
        value_net.eval()

        for _ in range(train_cfg.num_traversals_per_iter):
            state = initial_state_fn()

            # P1 관점 traversal
            traverse(
                state, traversing_player=0,
                adv_nets=adv_nets, value_net=value_net,
                adv_buffer=adv_buffers[0], strat_buffer=strat_buffer,
                iteration=t, max_depth=train_cfg.max_depth,
                device=device,
            )

            # P2 관점 traversal
            traverse(
                state, traversing_player=1,
                adv_nets=adv_nets, value_net=value_net,
                adv_buffer=adv_buffers[1], strat_buffer=strat_buffer,
                iteration=t, max_depth=train_cfg.max_depth,
                device=device,
            )

        # ── Training phase ──
        for player in range(2):
            if len(adv_buffers[player]) > 0:
                loss = train_advantage_net(
                    adv_nets[player], adv_buffers[player],
                    train_cfg, device, logger, iteration=t,
                )
                if logger:
                    logger.report_scalar(
                        "training", f"adv_loss_p{player+1}",
                        iteration=t, value=loss,
                    )

        # ── Logging ──
        if logger:
            logger.report_scalar("buffer", "advantage_p1", iteration=t, value=len(adv_buffers[0]))
            logger.report_scalar("buffer", "advantage_p2", iteration=t, value=len(adv_buffers[1]))
            logger.report_scalar("buffer", "strategy", iteration=t, value=len(strat_buffer))

        if t % 10 == 0:
            print(f"[Iter {t}/{train_cfg.num_cfr_iterations}] "
                  f"adv_buf=[{len(adv_buffers[0])}, {len(adv_buffers[1])}] "
                  f"strat_buf={len(strat_buffer)}")

        # ── Checkpoint ──
        if t % 100 == 0:
            torch.save({
                "iteration": t,
                "adv_net_0": adv_nets[0].state_dict(),
                "adv_net_1": adv_nets[1].state_dict(),
                "value_net": value_net.state_dict(),
            }, f"checkpoint_iter_{t}.pt")

    # ── Strategy network 학습 (최종) ──
    print(f"Strategy network 학습 시작 (버퍼: {len(strat_buffer)}개)...")
    strat_loss = train_strategy_net(
        strategy_net, strat_buffer, train_cfg, device,
        logger, num_epochs=20,
    )
    print(f"Strategy loss: {strat_loss:.6f}")

    torch.save(strategy_net.state_dict(), "strategy_net_final.pt")

    return strategy_net
