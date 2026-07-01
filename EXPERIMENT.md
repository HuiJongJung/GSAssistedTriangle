# GS-Assisted Triangle Splatting+ v3 (실험 정리)

> 이 문서는 "이 실험이 뭘 하려는지 + 코드가 원본을 어떻게 바꿨는지"를
> 이해하기 쉽게 정리한 것입니다. 정확한 값/스케줄은 마지막 절에 모아뒀습니다.
>
> **v3 변경 요지 (2026-07):** v2의 첫 결과(mipnerf360 bicycle)를 진단한 뒤 설계를
> 크게 바꿨습니다. 아래 "부록: v2 → v3 진단과 재설계"에 이유를 남겨둡니다.
> 한 줄로: **recruit 신호를 photometric → geometry evidence로**, **loss를 mixed →
> 분리형(decoupled)으로**, **GS 생애주기에 pruning을 추가**, **PSNR을 test셋 평균으로**.

## 0. 한 줄 요약

**가설:** "삼각형(triangle)이 잘 표현하지 못하는 구간은 학습도 잘 안 될 것이다."

**검증 방법:** 그 어려운 구간을 **임시 Gaussian(GS)** 으로 보조해서 학습시키고 →
GS가 **어디에 주로 모이는지** 확인하고 → 그 GS를 다시 **삼각형으로 변환**해서
최종 모델을 **삼각형만으로** 평가한다.

이렇게 하면 "삼각형의 한계가 *표현력* 문제인지, 아니면 *학습/초기화* 문제인지"를
가릴 수 있다.

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

**★ v3의 핵심 설계 — 삼각형은 A와 "완전히" 동일하게 학습된다:**
loss가 **분리형(decoupled)** 이라 GS의 gradient가 삼각형에 **절대 닿지 않는다**
(GS loss는 삼각형 텐서를 `detach`한 뒤 계산). 따라서 B의 삼각형 해(解)는
**설계상(by construction) baseline A와 동일**하다. 별도 ablation 없이 통제가 성립한다.
→ 이건 control로 반드시 확인한다 (§7-3).

**새로 추가한 것** (`gs_assisted/`):
- `gs_backend.py` — gsplat 래퍼 `GaussianBranch` (residual GS 브랜치, `prune()` 포함)
- `mixed_renderer/` — 삼각형+GS 합성 (`render_mixed`)
- `residual_gs/` — 삽입 정책(`policy.py`), **geometry 후보영역 검출**(`residual_mask.py`), 삽입(`insertion.py`)
- `compositing.py` — 합성 방식(over / depth_aware) + 기여율 계산
- `convert/geometry.py` — GS 1개 → 삼각형 2개 변환
- `diagnostics/`, `schedule.py` — 측정·저장 스케줄

> 설계 원칙: GPU 없이 검증 가능한 **순수 로직**(numpy/torch 공용)은 따로 빼서 CPU
> 단위테스트(43개)로 검증하고, GPU가 필요한 얇은 글루만 서버에서 돌린다.

## 3. GS를 "언제 / 어디에 / 무엇을" 넣는가 (삽입 기준)

**언제:** `gs_start_iter`(아래 기본값 참고) 이후, 저장 시점(`save_iters`)마다 시도.

**어디에 (후보 영역) — ★ v3에서 geometry evidence로 교체:** 아래 세 조건의 **교집합**
1. **광도 잔차(residual) 상위 10%** — 삼각형이 *색을 못 맞춘* 픽셀
2. **법선 불일치 상위 15%** — rendered normal이 metric3d prior와 어긋나는 픽셀
   (삼각형 *geometry가 틀린* 곳; 단순히 텍스처가 복잡한 곳이 아님)
3. **depth 불안정 상위 15%** — surf_depth의 국소 분산이 큰 픽셀 (edge·floater·불안정 geometry)

> **왜 바꿨나:** v2는 `residual ∩ (삼각형 alpha < 0.35)`였는데, Triangle Splatting+는
> opaque가 강제라 수렴하면 alpha≈1이 되어 이 게이트가 **사실상 죽는다**. 그래서 GS가
> 학습 초반 배경(늦게 수렴)에 잠깐 걸렸다가, 이후 삼각형이 덮으면 고아가 되어
> contribution≈0이 됐다("temporal recruitment mismatch"). geometry 신호는 alpha 포화
> 이후에도 살아있어 **진짜 geometry/topology failure**를 겨냥한다. (부록 참고)

**진짜로 넣을지 (지속성 게이트):**
- 같은 카메라(뷰)에서 그 영역이 `min_checkpoint_repeats`회 이상 반복해서 후보로 잡힌 곳만 채택.
- (서로 다른 뷰의 같은 픽셀좌표는 전혀 다른 3D 지점이라 비교 불가 → 카메라별로 이력 축적.)

**무엇을 넣나:**
- 채택된 영역에서 잔차 큰 픽셀들을 골라(용량 상한 내에서),
- 그 픽셀을 **삼각형 깊이로 3D 월드좌표에 역투영**한 위치에 GS를 생성.
- 초기값: 색 = 정답 이미지 픽셀, 불투명도 = 0.1, **크기 = init_scale(씬 비례 옵션)**, 회전 = 기본.

## 4. 어떻게 학습하는가 (분리형 손실 + 합성)

**합성 방식:** 기본 `depth_aware` — 삼각형과 GS의 **깊이를 비교**해 앞에 있는 걸 위로
얹는 2단 over 합성. (`over`로 바꾸면 깊이 무시)

**★ v3 손실 — 분리형(decoupled):**
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
- **v2의 mask 벌점은 제거**했다(낡은 accepted 마스크를 써서 좋은 자리로 옮긴 GS까지
  벌줬던 버그). 확산 억제는 sparsity + geometry gate + pruning이 담당.
- **법선 손실** = 원본 metric3d 법선 일치 (그대로).
- 삼각형과 GS는 **각자 옵티마이저로 따로** 업데이트.

**★ v3 GS 생애주기 (pruning):** 저장 시점마다, opacity가 `gs_prune_opacity` 아래로
붕괴한 GS를 **제거**한다. 삼각형이 그 영역을 접수했거나 애초에 진짜 geometry가
아니었다는 뜻이므로, "temporary holder"가 실제로 temporary하도록 청소한다.
(제거 후 GS 옵티마이저를 rebuild — Adam moment 정리)

## 5. GS → 삼각형 변환 (C)

1. B 체크포인트(`triangles/` + `gaussians.pt`) 로드
2. 각 GS를 **삼각형 패치로 변환** (GS 1개 → 삼각형 2개, 크기 보정 `size_factor`)
3. 변환된 삼각형을 원본 `TriangleModel`에 주입 → **삼각형만으로 finetune**
4. 최종 평가는 **GS 없이 삼각형 렌더만** (재현성 보장)

> ⚠️ C(convert)는 아직 v3 분리형에 맞춰 재정비하지 않았다. 분리형에선 삼각형이
> B에서 A와 동일하므로, **삼각형이 실제로 바뀌는 유일한 단계가 C다.** 다음 작업 후보.

## 6. 무엇을 보고 판단하나 (측정 / 시각화)

- **학습 중 자동 저장:** `diagnostics.json` — 숫자만
  (`metrics.psnr/ssim`, `triangle_count`, `gs_count`, `gs_contribution_ratio`, `wall_clock_s`,
  그리고 `eval` = {`triangle_only`, `mixed`})
  - **★ v3: PSNR/SSIM은 held-out test 카메라 전체를 렌더해 평균낸 값**이다.
    (v2는 랜덤 train 뷰 1장이라 노이즈·낙관·삽입타이밍에 튀었음.)
  - **headline `metrics.psnr` = 삼각형-only test PSNR** → A와 직접 비교 가능.
    `eval.mixed`는 GS 포함(T+G)일 때만 함께 저장(둘을 절대 안 섞음).
  - ⚠️ 이미지(PNG)는 자동 저장 안 됨.
- **시각화는 별도 스크립트:** `render_gs_assisted.py` 실행 →
  `T_only / G_only / T_plus_G / GS_red_overlay / alpha·depth 맵` PNG 생성.
  - **★ v3 `GS_red_overlay` 개선:** 모든 GS가 최소한(floor) 보이고, 진해도 씬이
    비쳐 보이도록(cap) 했다. 더는 "빨간 원으로 구역을 덮는" 마스크처럼 안 보임.
    `--red-gain/--red-cap/--red-floor`로 튜닝.

## 7. ★ 핵심적으로 볼 부분 (중요한 순서대로)

1. **`gs_contribution_ratio` — 실험의 생명줄.**
   이게 거의 0이면 "GS가 삼각형을 *못* 도왔다"는 뜻. v2에선 1.3e-6이었고, 원인은
   temporal recruitment mismatch로 규명됨(부록). v3 geometry gate + pruning이 이걸
   고쳤는지 반드시 추적.

2. **GS 분포 위치 = 연구 질문 그 자체.**
   새 `GS_red_overlay`로 확인: GS가 v2처럼 **배경 잎사귀(appearance)**로 가는가,
   아니면 v3 목표대로 **geometry가 틀린 곳(자전거 스포크·경계·topology 실패)**으로
   옮겨가는가?
   - geometry 쪽으로 가면 → gate 작동 = **논문 스토리 1번 성립.**
   - 여전히 잎사귀로 가면 → 그 영역 metric3d normal prior가 부실하다는 뜻이고, 그것도 발견.

3. **B-삼각형 ≡ A 확인 (control).**
   `--gs-start-iter 999999`로 GS를 끈 B를 A와 비교. 분리형이라 이제 GS가 삼각형을
   못 건드리므로 반드시 같아야 한다. 안 같으면 원인은 100% `_triangle_maintenance`
   port(아이디어 아님) → 그 블록을 upstream과 diff.

4. **변환 충실도 (C).**
   GS가 기여하던 표현이 삼각형 패치로 바뀐 뒤에도 유지되는가?
   헤드라인 비교는 **C(삼각형-only) vs A**, test셋 기준.

## 8. 알아둘 갭 / 주의

- **`configs/*.yaml`은 현재 학습에서 안 읽힘** — 정책은 CLI 인자 + 코드 기본값으로 정해짐.
- **C(convert)는 v3 분리형 미대응** (§5 주의).
- **init_scale 씬 비례는 opt-in** — `--gs-init-scale-frac 0`(기본)이면 절대값 `--gs-init-scale`
  (0.01) 사용. `>0`이면 `frac * scene_diag / count**(1/3)`.
- **depth/camera 규약은 정상** — v2 오버레이에서 GS가 예상 위치(게이트 걸린 곳)에 제대로
  렌더된 걸로 확인됨. (README의 'verify on server' 걱정은 해소.)
- `--gs-mask-weight`는 이제 **미사용**(분리형에서 stale-mask 벌점 제거).

---

## 부록: 기본 설정값

**저장 스케줄 (`save_iters`):**
30k 런 기준 `3000, 6000, ..., 30000` (10% 간격). `gs_start_iter`가 이 지점들에 없으면
진단용으로 추가 저장.

**GS 시작 시점 (`gs_start_iter`) 기본:** `max(5000, start_opacity_floor, start_pruning + 1000)`

**후보 영역 조건 (geometry gate):**
- 광도 잔차 상위 `residual_top_percent`(10%)
- 법선 불일치 상위 `normal_top_percent`(15%)
- depth 불안정 상위 `depth_top_percent`(15%)
- 같은 카메라에서 `min_checkpoint_repeats`(2)회 이상 반복

**용량 (씬 비례):**
- 전체 상한 = `clamp(gs_total_frac(0.10) * triangle_count, gs_min_total(2000), max_gs)`
- 1회 삽입 상한 = `max(gs_event_frac(0.02) * triangle_count, gs_min_event(500))`

**초기 GS:** opacity 0.1, 색 = GT 픽셀, 회전 identity,
scale = `gs_init_scale`(0.01) 또는 `gs_init_scale_frac`>0 시 씬 비례.

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

---

## 부록: v2 → v3 진단과 재설계 (기록)

**v2 첫 결과 (mipnerf360 bicycle) 진단:**
- `T_alpha`가 거의 전역 ≈1 (opaque 삼각형이 배경까지 덮음) → `alpha<0.35` 게이트가
  수렴 시점에 **죽음**.
- `GS_red_overlay`: GS가 전부 **상단 배경 수풀/나뭇잎 틈**(fuzzy appearance)에만 모임.
  자전거 스포크 같은 **얇은 geometry 실패엔 하나도 없음**(그곳은 alpha가 높아 게이트가
  못 걸림).
- `GS_alpha`/`G_only`가 매우 희미 → `gs_contribution_ratio ≈ 0`.

**원인 두 겹:**
1. **Temporal recruitment mismatch:** 배경은 늦게 수렴 → 초반 저-alpha일 때 게이트가
   걸려 GS 삽입 → 이후 삼각형이 alpha≈1로 덮음 → GS 고아·억압 → 기여 0.
2. **의미적 오조준:** photometric 신호라 승격하면 안 되는 fuzzy 잎사귀만 찾음.
   이는 wiki claim "high photometric residual ≠ geometry failure"의 예측과 정확히 일치.

**재설계(v3):**
- recruit: photometric+alpha → **geometry evidence(residual ∩ normal 불일치 ∩ depth 불안정)**.
- loss: mixed 공동학습 → **분리형**(삼각형 ≡ A, GS는 residual만).
- 생애주기: **GS pruning** 추가(붕괴 GS 제거).
- 하이퍼파라미터: **씬(삼각형 수) 비례** cap/scale.
- 측정: 랜덤 1뷰 → **test셋 전체 평균**, 삼각형-only와 mixed 분리 기록.
- 시각화: `GS_red_overlay`를 see-through로.
