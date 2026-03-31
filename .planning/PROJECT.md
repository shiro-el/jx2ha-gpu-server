# Pokemon Battle AI — Deep CFR

## What This Is

GPU 서버(RTX 5090)를 활용한 포켓몬 배틀 AI 시스템. Deep Counterfactual Regret Minimization(Deep CFR)을 사용하여 포켓몬 배틀을 POMDP + Stochastic Game으로 모델링하고, 최적 전략을 학습한다. Pokemon Showdown 6v6 Singles (OU) 래더에서 높은 레이팅 달성이 목표.

## Core Value

불완전정보 게임인 포켓몬 배틀에서 게임이론적으로 건전한(Nash 균형 수렴) AI 전략을 학습하는 것.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Deep CFR 알고리즘으로 포켓몬 배틀 전략 학습
- [ ] POMDP + Stochastic Game으로 배틀 상태 모델링
- [ ] poke-env을 통한 Pokemon Showdown 연동
- [ ] PokeAPI에서 전 세대 포켓몬 데이터 수집 및 관리
- [ ] 6v6 Singles (OU 포맷) 지원
- [ ] RTX 5090 GPU를 활용한 학습 파이프라인
- [ ] Showdown 래더에서 자동 대전 및 레이팅 추적
- [ ] 학습 과정 모니터링 및 실험 추적

### Out of Scope

- 더블 배틀 — 상태 공간이 기하급수적으로 증가, v1에서 제외
- 3v3 포맷 — 6v6 OU에 집중
- LLM 기반 전략 판단 — 순수 RL/게임이론 접근
- 웹/앱 UI — 연구 목적이므로 CLI/스크립트 기반
- 팀 빌딩 자동화 — v1에서는 배틀 전략에 집중

## Context

- **서버 환경**: ClearML GPU 서버, RTX 5090
- **게임이론 배경**: 포켓몬 배틀은 simultaneous move (양쪽이 동시에 기술 선택), 불완전 정보 (상대 기술/아이템/노력치 모름), 확률적 (명중률, 급소, 추가효과)인 2인 제로섬 게임
- **Deep CFR**: Brown et al. (2019)의 Deep Counterfactual Regret Minimization — 불완전정보 게임에서 Nash 균형으로 수렴하는 딥러닝 기반 CFR
- **poke-env**: Python 라이브러리로 Pokemon Showdown 서버와 연동하여 RL 학습 환경 제공
- **연구 목적**: 논문 작성 가능한 수준의 실험 설계 및 결과 분석 포함

## Constraints

- **Compute**: RTX 5090 1대 — 학습 시간 및 메모리 한계 고려 필요
- **환경**: poke-env + Pokemon Showdown 서버 의존 — 서버 안정성 이슈 가능
- **상태 공간**: 전 세대 포켓몬 (1000+ 종, 800+ 기술) — 상태 표현 및 추상화 전략 필수
- **CFR 확장성**: vanilla CFR은 큰 게임 트리에서 비실용적 — Deep CFR + 상태 추상화 필요

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Deep CFR 선택 | 불완전정보 게임에서 Nash 균형 수렴 보장, PPO 대비 이론적 우위 | — Pending |
| poke-env 사용 | 검증된 Showdown 연동 라이브러리, RL 학습 환경 표준 | — Pending |
| 6v6 Singles OU | Showdown에서 가장 활발한 포맷, 래더 레이팅으로 객관적 평가 가능 | — Pending |
| PokeAPI 데이터 | 공개 REST API, 전 세대 커버리지 | — Pending |
| 전 세대 포켓몬 | 연구 완성도를 위해 전체 포켓몬 포함, 추상화로 복잡도 관리 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-31 after initialization*
