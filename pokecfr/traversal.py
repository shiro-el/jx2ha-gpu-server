"""Deep CFR traversal — poke-engine 기반.

poke-engine의 generate_instructions()가 chance 노드를 확률별로 완전 열거하므로
Monte Carlo 샘플링 없이 정확한 기대값 계산이 가능.

한 턴 구조:
    P1 행동 × P2 행동 → generate_instructions → 확률 분기들
    → 각 분기에 apply → 재귀 → reverse
"""

import torch
import torch.nn.functional as F

from poke_engine import State as PEState
from poke_engine import generate_instructions, get_all_options, is_terminal

from pokecfr.buffers import AdvantageBuffer, StrategyBuffer
from pokecfr.converter import state_to_infoset
from pokecfr.networks import AdvantageNetwork, ValueNetwork


def regret_matching(regrets: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Regret matching: 양수 regret에 비례한 전략 생성."""
    positive = F.relu(regrets) * action_mask
    total = positive.sum()
    if total > 0:
        return positive / total
    return action_mask / action_mask.sum()


def _parse_options(options: list[str]) -> tuple[list[str], list[str]]:
    """poke-engine 옵션 → (기술 리스트, 교체 리스트) 분리.

    get_all_options은 "switch <index>" 형식으로 반환 (인덱스 기반).
    기술은 tera/mega 제외. "none"은 무시.
    """
    moves = []
    switches = []
    seen_moves = set()

    for opt in options:
        if opt == "none":
            continue
        elif opt.startswith("switch"):
            switches.append(opt)
        elif "-tera" not in opt and "-mega" not in opt:
            if opt not in seen_moves:
                seen_moves.add(opt)
                moves.append(opt)

    return moves, switches


def _options_to_mask(options: list[str], num_actions: int = 9) -> torch.Tensor:
    """poke-engine 옵션 리스트 → action mask 텐서.

    인덱싱: 0~3 기술, 4~8 교체 (벤치 순서대로)
    """
    moves, switches = _parse_options(options)
    mask = torch.zeros(num_actions, dtype=torch.float32)

    for i in range(min(len(moves), 4)):
        mask[i] = 1.0
    for i in range(min(len(switches), 5)):
        mask[4 + i] = 1.0

    return mask


def _action_idx_to_option(idx: int, options: list[str]) -> str:
    """action index → poke-engine 옵션 문자열."""
    moves, switches = _parse_options(options)

    if idx < 4 and idx < len(moves):
        return moves[idx]
    elif idx >= 4 and (idx - 4) < len(switches):
        return switches[idx - 4]
    return moves[0] if moves else switches[0]


def _add_batch_dim(data: dict, device: torch.device | None = None) -> dict:
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _add_batch_dim(v, device)
        elif isinstance(v, torch.Tensor):
            t = v.unsqueeze(0)
            result[k] = t.to(device) if device is not None else t
        else:
            result[k] = v
    return result


@torch.no_grad()
def traverse(
    state: PEState,
    traversing_player: int,
    adv_nets: list[AdvantageNetwork],
    value_net: ValueNetwork,
    adv_buffer: AdvantageBuffer,
    strat_buffer: StrategyBuffer,
    iteration: int,
    max_depth: int,
    turn: int = 0,
    reach_probs: list[float] | None = None,
    device: torch.device | None = None,
) -> float:
    """External sampling Deep CFR traversal (poke-engine 기반).

    양쪽 행동을 동시에 처리:
    - traversing player: 모든 행동 탐색
    - opponent: 전략에서 행동 샘플링

    각 (행동_trav, 행동_opp) 쌍에 대해 generate_instructions로
    확률 분기를 열거하고, 확률 가중 기대값을 계산.

    Returns:
        traversing_player 관점의 counterfactual value
    """
    if reach_probs is None:
        reach_probs = [1.0, 1.0]

    # ── Terminal 체크 ──
    terminal, winner = is_terminal(state)
    if terminal:
        if winner == "tie":
            return 0.0
        p1_wins = winner == "side_one"
        if traversing_player == 0:
            return 1.0 if p1_wins else -1.0
        else:
            return -1.0 if p1_wins else 1.0

    # ── Depth limit → value net 평가 ──
    if turn >= max_depth:
        infoset = state_to_infoset(state, traversing_player, turn)
        batched = _add_batch_dim(infoset, device)
        return value_net(batched).item()

    # ── 양쪽 옵션 가져오기 ──
    s1_opts, s2_opts = get_all_options(state)

    # "none"만 있으면 pass 처리 — force_switch에서 한쪽만 행동
    s1_has_actions = any(o != "none" for o in s1_opts)
    s2_has_actions = any(o != "none" for o in s2_opts)

    if not s1_has_actions and not s2_has_actions:
        return 0.0

    s1_mask = _options_to_mask(s1_opts)
    s2_mask = _options_to_mask(s2_opts)
    if device is not None:
        s1_mask = s1_mask.to(device)
        s2_mask = s2_mask.to(device)

    # ── 각 플레이어의 전략 계산 ──
    trav = traversing_player
    opp = 1 - trav
    trav_opts = s1_opts if trav == 0 else s2_opts
    opp_opts = s2_opts if trav == 0 else s1_opts
    trav_mask = s1_mask if trav == 0 else s2_mask
    opp_mask = s2_mask if trav == 0 else s1_mask

    # Traversing player 전략
    trav_infoset = state_to_infoset(state, trav, turn)
    trav_batched = _add_batch_dim(trav_infoset, device)
    trav_regrets = adv_nets[trav](trav_batched).squeeze(0)
    trav_strategy = regret_matching(trav_regrets, trav_mask)

    # Opponent 전략
    opp_infoset = state_to_infoset(state, opp, turn)
    opp_batched = _add_batch_dim(opp_infoset, device)
    opp_regrets = adv_nets[opp](opp_batched).squeeze(0)
    opp_strategy = regret_matching(opp_regrets, opp_mask)

    # Strategy buffer에 상대 전략 저장
    state_str = state.to_string()
    strat_buffer.store(state_str, opp, turn, opp_strategy, opp_mask, iteration, reach_probs[opp])

    # ── Opponent 행동 샘플링 (external sampling) ──
    opp_has_actions = any(o != "none" for o in opp_opts)
    if opp_has_actions and opp_mask.sum() > 0:
        opp_action_idx = torch.multinomial(opp_strategy, 1).item()
        opp_move = _action_idx_to_option(opp_action_idx, opp_opts)
        new_reach_opp = reach_probs.copy()
        new_reach_opp[opp] *= opp_strategy[opp_action_idx].item()
    else:
        opp_move = "none"
        new_reach_opp = reach_probs.copy()

    # ── Traversing player: 모든 행동 탐색 ──
    trav_has_actions = any(o != "none" for o in trav_opts)
    if not trav_has_actions or trav_mask.sum() == 0:
        # Traversing player에게 선택지 없음 (상대만 force_switch)
        # 상대 행동을 하나 골라서 진행
        if opp_has_actions:
            opp_action_idx = torch.multinomial(opp_strategy, 1).item()
            opp_move_fs = _action_idx_to_option(opp_action_idx, opp_opts)
        else:
            opp_move_fs = "none"
        if trav == 0:
            branches = generate_instructions(state, "none", opp_move_fs)
        else:
            branches = generate_instructions(state, opp_move_fs, "none")
        ev = 0.0
        for branch in branches:
            prob = branch.percentage / 100.0
            if prob <= 0:
                continue
            next_state = state.apply_instructions(branch)
            ev += prob * traverse(
                next_state, trav, adv_nets, value_net,
                adv_buffer, strat_buffer, iteration, max_depth,
                turn + 1, reach_probs, device,
            )
        return ev

    num_actions = trav_mask.shape[0]
    action_values = torch.zeros(num_actions, dtype=torch.float32, device=device)

    for a_idx in range(num_actions):
        if trav_mask[a_idx] == 0:
            continue

        trav_move = _action_idx_to_option(a_idx, trav_opts)

        # poke-engine에 양쪽 행동 전달
        if trav == 0:
            s1_move, s2_move = trav_move, opp_move
        else:
            s1_move, s2_move = opp_move, trav_move

        # 확률 분기 열거
        branches = generate_instructions(state, s1_move, s2_move)

        # 각 분기에 대해 확률 가중 재귀
        ev = 0.0
        for branch in branches:
            prob = branch.percentage / 100.0
            if prob <= 0:
                continue

            next_state = state.apply_instructions(branch)
            branch_value = traverse(
                next_state, trav, adv_nets, value_net,
                adv_buffer, strat_buffer, iteration, max_depth,
                turn + 1, new_reach_opp, device,
            )
            ev += prob * branch_value

        action_values[a_idx] = ev

    # ── Counterfactual regret 계산 ──
    trav_ev = (trav_strategy * action_values).sum().item()
    cfr_regret = action_values - trav_ev

    # Advantage buffer에 저장
    adv_buffer.store(state_str, trav, turn, cfr_regret, trav_mask, iteration)

    return trav_ev
