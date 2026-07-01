# 2026-07-01 · v3 — 분리형 loss + geometry-evidence recruitment + GS 생애주기

commit `f873871` (main). 이전 상태: v2 (`45d2c98`).

## 트리거

v2 첫 정식 결과(mipnerf360 `bicycle`, `diag_view0/`)를 진단.

## 진단 (결과가 말해준 것)

네 진단 이미지가 한 방향을 가리켰다.

- **`T_alpha` ≈ 전역 1.0** — Triangle Splatting+는 opaque가 강제라 수렴하면 배경까지
  삼각형이 불투명하게 덮는다. → 삽입 게이트의 `alpha < 0.35` 조건이 **수렴 시점에 죽는다**.
- **`GS_red_overlay`** — GS가 전부 **상단 배경 수풀/나뭇잎 틈**(fuzzy appearance)에만 모임.
  자전거 스포크 같은 **얇은 geometry 실패엔 하나도 없음**(그곳은 alpha가 높아 게이트 미발동).
- **`GS_alpha` / `G_only`** 매우 희미 → `gs_contribution_ratio ≈ 0` (v2 스모크 1.3e-6).

**원인 두 겹:**
1. **Temporal recruitment mismatch** — 배경은 늦게 수렴 → 초반 저-alpha일 때 게이트가
   걸려 GS 삽입 → 이후 삼각형이 alpha≈1로 덮음 → GS 고아·억압 → 기여 0.
2. **의미적 오조준** — photometric 신호라 승격하면 안 되는 fuzzy 잎사귀만 찾음. 이는 wiki
   claim "high photometric residual ≠ geometry failure"의 예측과 정확히 일치. 즉 실패가
   가설의 증거였다.

부수 확인: GS가 게이트 걸린 위치에 제대로 렌더됨 → **depth/camera 규약은 정상**
(README의 'verify on server' 걱정 해소).

## 결정 (사용자 선택)

- **loss 구조:** 분리형(decoupled). [삼각형 ≡ A 보장 + GS는 순수 진단자]
- **recruit 신호:** geometry evidence 추가 (normal 불일치 + depth 불안정).
- (함께) GS 생애주기 pruning 추가, 하이퍼파라미터 씬 비례, PSNR test셋 평균, overlay 개선.

## 코드 변경점 (파일별)

- `residual_gs/residual_mask.py`
  - `normal_disagreement()` 추가 — `0.5*(1-cos)`, metric3d prior 대비 법선 어긋남.
  - `geometry_candidate_mask()` 추가 — residual ∩ normal-불일치 ∩ depth-불안정(선택),
    전부 top-percent라 scale-free.
  - 기존 `candidate_mask`(alpha 게이트)는 **삭제 안 함** — 참조/테스트/baseline용.
- `residual_gs/policy.py` — `normal_top_percent`(15), `depth_top_percent`(15) 필드 추가.
- `gs_backend.py` — `GaussianBranch.prune(keep_mask)` 추가.
- `train_gs_assisted.py`
  - normal prior를 렌더 직후로 끌어올려 gate·법선손실 공용.
  - gate를 `geometry_candidate_mask` + `_depth_instability()`(avg_pool 국소분산)로 교체.
  - **분리형 loss:** `loss_tri = image(T,gt)+법선+weight` (A와 동일),
    `loss_gs = image(composite(T.detach(),G),gt) + sparse*opacity.mean()`. stale-mask 벌점 제거.
  - save마다 opacity < `gs_prune_opacity`(0.01) GS pruning + 옵티마이저 rebuild.
  - `_dynamic_policy()`(cap 비례) + `_resolve_init_scale()`(scale 비례) + 새 CLI 인자.
  - `_evaluate_testset()` — held-out test 카메라 전체 평균 PSNR/SSIM (삼각형-only headline,
    mixed 별도). 단일 랜덤뷰 폐기.
- `render_gs_assisted.py` — `GS_red_overlay` see-through 재작성(floor/cap/gain),
  `_red_branch` 제거. `--red-gain/-cap/-floor` 추가.

## 검증

- CPU 단위테스트 43/43 통과. 새 순수 함수 numpy 스모크 OK. 전 파일 `py_compile` OK.
- GPU 경로는 미검증 → 서버 스모크 필요.

## 서버에서 확인할 것

1. control(`--gs-start-iter 999999`)로 **B삼각형 ≡ A** (분리형 통제 성립 / port 격리).
2. 새 overlay에서 GS가 잎사귀 대신 **geometry 실패(스포크·경계)**로 옮겨가는가.
3. `gs_contribution_ratio`가 0을 벗어나는가.
4. 헤드라인 **C vs A** (test셋).

여전히 잎사귀로 가면 → 그 영역 metric3d normal prior 부실이라는 또 다른 발견.

## 미해결 / 다음

- C(convert)는 아직 분리형 미대응 — 분리형에선 삼각형이 실제 바뀌는 유일한 단계라 우선순위.
- `configs/*.yaml`은 학습에서 미사용(CLI만).
