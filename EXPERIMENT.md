# GS-Assisted Triangle Splatting+ (실험 설계)

> 이 문서는 **현재 실험 설계 스냅샷**이다 — "지금 코드가 무엇을 하는가".
> 설계 변천사(왜 이렇게 됐나 / 무엇을 바꿨나)는 [`design_log/`](design_log/)에 날짜별로 있다.

## 0. 한 줄 요약

**가설:** "삼각형(triangle)이 잘 표현하지 못하는 구간은 학습도 잘 안 될 것이다."

**검증 방법:** 그 어려운 구간을 **임시 Gaussian(GS)** 으로 보조해서 학습시키고 →
GS가 **어디에 주로 모이는지** 확인하고 → 그 GS를 다시 **삼각형으로 변환**해서
최종 모델을 **삼각형만으로** 평가한다.

이렇게 하면 "삼각형의 한계가 *표현력* 문제인지, 아니면 *학습/초기화* 문제인지"를 가릴 수 있다.

## 1. 세 가지 변형 (A / B / C)

| 변형 | 코드 | 역할 |
|---|---|---|
| **A** (baseline) | 원본 `third_party/.../train.py` (수정 안 함) | 순수 Triangle Splatting+. 비교 기준선. |
| **B** (mixed) | `gs_assisted/train_gs_assisted.py` | 삼각형 + **임시 residual GS** 공동 학습. **실험의 본체.** |
| **C** (convert) | `gs_assisted/convert_gs_to_triangles.py` | GS → 삼각형 패치로 변환 후 **삼각형만으로** finetune/평가. |

출력은 전부 한 뿌리 아래에 분리 저장:
`outputs/gs_assisted_triangle_plus_v2/<dataset>/<scene>/<variant>/`

## 2. 원본을 어떻게 바꿨나

**원본 그대로 빌려 쓰는 것** (건드리지 않음):
- 삼각형 모델·렌더러: `scene.TriangleModel`, `triangle_renderer.render`
- 삼각형 하이퍼파라미터 전부 (densify·prune·opacity 스케줄·learning rate):
  원본 인자 파서를 **빈 인자로 파싱해 기본값 그대로** 가져옴
- 삼각형 유지보수(prune/densify), 법선 손실(metric3d), SH 스케줄 → 원본 train.py 블록을 **그대로 포팅**

**★ 핵심 설계 — 삼각형은 A와 "완전히" 동일하게 학습된다:**
loss가 **분리형(decoupled)** 이라 GS의 gradient가 삼각형에 **절대 닿지 않는다**
(GS loss는 삼각형 텐서를 `detach`한 뒤 계산). 따라서 B의 삼각형 해(解)는
**설계상(by construction) baseline A와 동일**하다. 별도 ablation 없이 통제가 성립한다.
→ control로 반드시 확인한다 (§7-3).

**새로 추가한 것** (`gs_assisted/`):
- `gs_backend.py` — gsplat 래퍼 `GaussianBranch` (residual GS 브랜치, `prune()` 포함)
- `mixed_renderer/` — 삼각형+GS 합성 (`render_mixed`)
- `residual_gs/` — 삽입 정책(`policy.py`), geometry 후보영역 검출(`residual_mask.py`), 삽입(`insertion.py`)
- `compositing.py` — 합성 방식(over / depth_aware) + 기여율 계산
- `convert/geometry.py` — GS 1개 → 삼각형 2개 변환
- `diagnostics/`, `schedule.py` — 측정·저장 스케줄

> 설계 원칙: GPU 없이 검증 가능한 **순수 로직**(numpy/torch 공용)은 따로 빼서 CPU
> 단위테스트(43개)로 검증하고, GPU가 필요한 얇은 글루만 서버에서 돌린다.

## 3. GS를 "언제 / 어디에 / 무엇을" 넣는가 (삽입 기준)

**언제:** `gs_start_iter`(아래 기본값 참고) 이후, 저장 시점(`save_iters`)마다 시도.

**어디에 (후보 영역) — geometry-failure evidence의 교집합:**
1. **광도 잔차(residual) 상위 10%** — 삼각형이 *색을 못 맞춘* 픽셀
2. **법선 불일치 상위 15%** — rendered normal이 metric3d prior와 어긋나는 픽셀
   (삼각형 *geometry가 틀린* 곳; 단순히 텍스처가 복잡한 곳이 아님)
3. **depth 불안정 상위 15%** — surf_depth의 국소 분산이 큰 픽셀 (edge·floater·불안정 geometry)

> 왜 alpha가 아니라 geometry 신호인가: Triangle Splatting+는 opaque가 강제라 수렴하면
> alpha≈1이 되어 "덜 덮인 곳" 신호가 사라진다. geometry 신호는 그 이후에도 살아있어
> **진짜 geometry/topology failure**를 겨냥하고, 승격하면 안 되는 fuzzy appearance(잎사귀
> 등)를 피한다.

**진짜로 넣을지 (지속성 게이트):**
- 같은 카메라(뷰)에서 그 영역이 `min_checkpoint_repeats`회 이상 반복해서 후보로 잡힌 곳만 채택.
- (서로 다른 뷰의 같은 픽셀좌표는 전혀 다른 3D 지점이라 비교 불가 → 카메라별로 이력 축적.)

**무엇을 넣나:**
- 채택된 영역에서 잔차 큰 픽셀들을 골라(용량 상한 내에서),
- 그 픽셀을 **삼각형 깊이로 3D 월드좌표에 역투영**한 위치에 GS를 생성.
- 초기값: 색 = 정답 이미지 픽셀, 불투명도 = 0.1, 크기 = `init_scale`(씬 비례 옵션), 회전 = 기본.

## 4. 어떻게 학습하는가 (분리형 손실 + 합성)

**합성 방식:** 기본 `depth_aware` — 삼각형과 GS의 **깊이를 비교**해 앞에 있는 걸 위로
얹는 2단 over 합성. (`over`로 바꾸면 깊이 무시)

**손실 — 분리형(decoupled):**
```
loss = loss_tri + loss_gs

loss_tri = image손실(T, gt)          # ← baseline A와 완전히 동일 (삼각형만)
         + 법선손실(T)
         + 삼각형 weight손실(초반)

loss_gs  = image손실(composite(T.detach(), G), gt)   # ← GS가 residual만 학습
         + gs_sparse_weight * opacity평균             # 가벼운 sparsity
         (gs가 존재할 때만; 아니면 0)
```
- **삼각형 image 손실**은 항상 **삼각형-only 렌더** 기준. GS 켜진 뒤에도 안 바뀜 → 삼각형 ≡ A.
- **GS image 손실**은 삼각형을 `detach`한 합성 기준 → gradient가 GS로만 흐름.
  GS는 gate가 넣어준 자리에서만 존재하므로, **gate가 위치를, 이 손실이 채움의 질을** 결정.
- **법선 손실** = 원본 metric3d 법선 일치 (그대로).
- 삼각형과 GS는 **각자 옵티마이저로 따로** 업데이트.

**GS 생애주기 (pruning):** 저장 시점마다, opacity가 `gs_prune_opacity` 아래로 붕괴한 GS를
**제거**한다. 삼각형이 그 영역을 접수했거나 애초에 진짜 geometry가 아니었다는 뜻이므로,
"temporary holder"가 실제로 temporary하도록 청소한다. (제거 후 GS 옵티마이저 rebuild.)

## 5. GS → 삼각형 변환 (C)

1. B 체크포인트(`triangles/` + `gaussians.pt`) 로드
2. 각 GS를 **삼각형 패치로 변환** (GS 1개 → 삼각형 2개, 크기 보정 `size_factor`)
3. 변환된 삼각형을 원본 `TriangleModel`에 주입 → **삼각형만으로 finetune**
4. 최종 평가는 **GS 없이 삼각형 렌더만** (재현성 보장)

> ⚠️ C는 아직 분리형에 맞춰 재정비하지 않았다. 분리형에선 삼각형이 B에서 A와 동일하므로,
> **삼각형이 실제로 바뀌는 유일한 단계가 C다.** 다음 작업 후보.

## 6. 무엇을 보고 판단하나 (측정 / 시각화)

- **학습 중 자동 저장:** `diagnostics.json` — 숫자만
  (`metrics.psnr/ssim`, `triangle_count`, `gs_count`, `gs_contribution_ratio`, `wall_clock_s`,
  `eval` = {`triangle_only`, `mixed`})
  - **PSNR/SSIM은 held-out test 카메라 전체를 렌더해 평균낸 값**이다.
  - **headline `metrics.psnr` = 삼각형-only test PSNR** → A와 직접 비교 가능.
    `eval.mixed`는 GS 포함(T+G)일 때만 함께 저장(둘을 절대 안 섞음).
  - ⚠️ 이미지(PNG)는 자동 저장 안 됨.
- **시각화는 별도 스크립트:** `render_gs_assisted.py` 실행 →
  `T_only / G_only / T_plus_G / GS_red_overlay / alpha·depth 맵` PNG 생성.
  - **`GS_red_overlay`** = 삼각형 렌더 위에 GS 위치를 빨갛게. 모든 GS가 최소한(floor) 보이고,
    진해도 씬이 비쳐 보이도록(cap) 함 → "GS가 어디 모이나"가 이 실험의 핵심 그림.
    `--red-gain/--red-cap/--red-floor`로 튜닝.

## 7. ★ 핵심적으로 볼 부분 (중요한 순서대로)

1. **`gs_contribution_ratio` — 실험의 생명줄.**
   거의 0이면 "GS가 삼각형을 *못* 도왔다"는 뜻이라 가설 검증 자체가 안 됨. 본 실험에서 반드시 추적.

2. **GS 분포 위치 = 연구 질문 그 자체.**
   `GS_red_overlay`로 확인: GS가 **geometry가 틀린 곳(스포크·경계·topology 실패)**으로 모이는가,
   아니면 **fuzzy appearance(배경 잎사귀)**로 새는가?
   - geometry 쪽 = geometry gate 작동 = 연구 스토리 성립.
   - appearance 쪽 = 그 영역 metric3d normal prior가 부실하다는 뜻(그것도 발견).

3. **B-삼각형 ≡ A 확인 (control).**
   `--gs-start-iter 999999`로 GS를 끈 B를 A와 비교. 분리형이라 반드시 같아야 한다.
   안 같으면 원인은 100% `_triangle_maintenance` port(아이디어 아님) → upstream과 diff.

4. **변환 충실도 (C).**
   GS가 기여하던 표현이 삼각형 패치로 바뀐 뒤에도 유지되는가?
   헤드라인 비교는 **C(삼각형-only) vs A**, test셋 기준.

## 8. 알아둘 갭 / 주의

- **`configs/*.yaml`은 현재 학습에서 안 읽힘** — 정책은 CLI 인자 + 코드 기본값으로 정해짐.
- **C(convert)는 아직 분리형 미대응** (§5 주의).
- **init_scale 씬 비례는 opt-in** — `--gs-init-scale-frac 0`(기본)이면 절대값 `--gs-init-scale`
  (0.01) 사용. `>0`이면 `frac * scene_diag / count**(1/3)`.
- `--gs-mask-weight`는 이제 **미사용**(분리형에서 stale-mask 벌점 제거).

---

## 부록: 기본 설정값

**저장 스케줄 (`save_iters`):**
30k 런 기준 `3000, 6000, ..., 30000` (10% 간격). `gs_start_iter`가 이 지점들에 없으면 진단용으로 추가 저장.

**GS 시작 시점 (`gs_start_iter`) 기본:** `max(5000, start_opacity_floor, start_pruning + 1000)`

**후보 영역 조건 (geometry gate):**
- 광도 잔차 상위 `residual_top_percent`(10%)
- 법선 불일치 상위 `normal_top_percent`(15%)
- depth 불안정 상위 `depth_top_percent`(15%)
- 같은 카메라에서 `min_checkpoint_repeats`(2)회 이상 반복

**용량 (씬 비례):**
- 전체 상한 = `clamp(gs_total_frac(0.10) * triangle_count, gs_min_total(2000), max_gs)`
- 1회 삽입 상한 = `max(gs_event_frac(0.02) * triangle_count, gs_min_event(500))`

**초기 GS:** opacity 0.1, 색 = GT 픽셀, 회전 identity, scale = `gs_init_scale`(0.01) 또는 `gs_init_scale_frac`>0 시 씬 비례.

**손실 가중치:** image = upstream `lambda_dssim` 혼합, `gs_sparse_weight`(0.001).

**GS pruning:** 저장 시점마다 opacity < `gs_prune_opacity`(0.01) 제거.

**평가:** held-out test 카메라 전체 평균(PSNR/SSIM). `--eval-max-views 0` = 전체.

**저장 진단 항목:**
`metrics.psnr/ssim`(삼각형-only test 평균), `eval.{triangle_only, mixed}`,
`triangle_count`, `gs_count`, `gs_contribution_ratio`, `wall_clock_s`.
별도 렌더: `T_only / G_only / T_plus_G / GS_red_overlay`, 삼각형/GS의 alpha·depth 맵.

**합격 기준:**
- A와 B/C 코드 경로가 분리돼 있음
- A/B/C 출력이 같은 루트 아래 분리됨
- B는 (control 아닐 때) 삽입 후 0이 아닌 GS 수 + GS 기여율 기록
- B의 삼각형-only test PSNR ≈ A (분리형 통제 성립)
- C 최종이 삼각형 렌더만으로 재현 가능
