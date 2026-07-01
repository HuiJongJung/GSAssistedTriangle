# 2026-07-01 · C 변환을 선택적·rank-aware 승격으로 + 주입 patch 보호 + A/B/C 렌더

v3 재설계([2026-07-01-v3](2026-07-01-v3-decoupled-geometry-gate.md)) 직후, C(convert)를 손봤다.

## 트리거

"gs→triangle 변환을 마지막에 한 방에 하고 마는데, 최적화가 필요하지 않나"라는 질문.
그리고 파이프라인이 C를 렌더하지 않아 **변환 결과를 눈으로 볼 방법이 없었다.**

## 진단

- C가 **모든 GS를 무차별 변환** → wiki claim(evidence-based promotion, 승격할 것만)과 어긋남.
- GS→평면 quad는 **surface-like(rank≈2)** 에만 맞음. needle(rank1)/blob(rank3, fuzzy)엔 부적합.
- 주입 patch의 vertex_weight를 GS opacity(~0.1)로 세팅 + finetune이 importance bookkeeping을
  갱신 안 함 → **첫 maintenance(iter 500)에서 방금 변환한 걸 prune할 위험.**
- C의 최종 PSNR이 **단일 train view**라 A와 비교 불가였음.

## 결정 (사용자 선택, 전부 추천안)

- 승격 gate: **opacity 임계값만** (변환 시점 gaussians.pt에 바로 있음. contribution/재출현
  이력은 학습 때 미저장이라 지금은 불가 — 후속 후보).
- rank 라우팅: **surface(rank≈2)만 quad, needle/blob 제외**.
- 주입 보호: **finetune 초반 pruning 면제**.

## 코드 변경점

- `convert_gs_to_triangles.py`
  - `_promotion_mask()` 추가 — confident(opacity≥th) ∩ surface-like(s2/s1≥`surface_ratio`
    AND s3/s1≤`flat_ratio`). rank는 scales(=std, 정렬)로 판정(covariance eigenvalue=std²).
  - `_inject_converted_triangles(..., keep)` — 승격된 GS만 변환·주입, `(before,after)` 반환.
  - finetune 루프: 매 iter **image_size/importance_score bookkeeping 갱신**(training_b 미러),
    `it > protect_iters`에서만 `_triangle_maintenance` 호출.
  - 최종 eval을 `_evaluate_testset`(test셋 평균)으로 — A/B와 like-for-like.
  - `summary.json`에 `promotion`(n_total/surface_like/confident/promoted) 기록.
  - 새 인자: `--opacity-threshold`(0.2) `--surface-ratio`(0.3) `--flat-ratio`(0.3)
    `--protect-iters`(1000) `--eval-max-views`.
- `scripts/run_scene.sh`
  - 렌더 단계를 C 뒤로 옮기고 **B와 C를 둘 다** 렌더 → `outputs/diag/<scene>/{B,C}`.
    (A는 upstream 저장 포맷이 우리 `TriangleModel`과 달라 못 읽음 → 분리형 덕에 **B의
    `T_only`가 baseline 대용**, 엄밀 A≡B는 `--control`이 PSNR로 잡음.)

## 검증

- CPU 43 테스트 통과, `py_compile`/`bash -n` OK. `_promotion_mask` 로직 수기 검증
  (surface 통과, needle/blob/저opacity 제외). torch 경로는 서버 스모크 필요.

## 서버에서 확인할 것

- `summary.json`의 `promotion`: 몇 개가 surface_like/confident/promoted 됐나 (0이면 임계값 과함).
- `triangles_before/after_conversion`: 주입 patch가 finetune 후 살아남았나(즉시 prune 여부).
- `outputs/diag/<scene>/C/T_only.png` vs `B/T_only.png`: 변환된 mesh가 baseline 대비 나아졌나.

## 미해결 / 다음

- 승격 gate에 contribution/gate-재출현을 쓰려면 학습 때 per-GS 메타데이터 저장 필요.
- needle→strip 변환(현재 needle 제외). progressive(학습 중) 승격은 그다음 단계.
- C finetune이 protect 이후 densify도 함(iteration 스케줄) — 필요 시 정지 고려.
