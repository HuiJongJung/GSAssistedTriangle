# GS-Assisted Triangle Splatting+ v2 (실험 정리)

> 이 문서는 "이 실험이 뭘 하려는지 + 코드가 원본을 어떻게 바꿨는지"를
> 이해하기 쉽게 정리한 것입니다. 정확한 값/스케줄은 마지막 절에 모아뒀습니다.

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
  → 그래서 B의 삼각형 쪽은 baseline(A)과 **완전히 같은 조건**에서 돈다.
- 삼각형 유지보수(prune/densify), 법선 손실(metric3d), SH 스케줄 → 원본 train.py 블록을 **그대로 포팅**

**새로 추가한 것** (`gs_assisted/`):
- `gs_backend.py` — gsplat 래퍼 `GaussianBranch` (= residual GS 브랜치)
- `mixed_renderer/` — 삼각형+GS 합성 (`render_mixed`)
- `residual_gs/` — 삽입 정책(`policy.py`), 후보영역 검출(`residual_mask.py`), 삽입(`insertion.py`)
- `compositing.py` — 합성 방식(over / depth_aware) + 기여율 계산
- `convert/geometry.py` — GS 1개 → 삼각형 2개 변환
- `diagnostics/`, `schedule.py` — 측정·저장 스케줄

> 설계 원칙: GPU 없이 검증 가능한 **순수 로직**(numpy/torch 공용)은 따로 빼서 CPU
> 단위테스트(43개)로 검증하고, GPU가 필요한 얇은 글루만 서버에서 돌린다.

## 3. GS를 "언제 / 어디에 / 무엇을" 넣는가 (삽입 기준)

**언제:** `gs_start_iter`(아래 기본값 참고) 이후, 저장 시점(`save_iters`)마다 시도.

**어디에 (후보 영역):** 아래 두 조건의 **교집합**
1. **광도 잔차(residual) 상위 10%** — 삼각형이 *색을 못 맞춘* 픽셀
2. **삼각형 alpha < 0.35** — 삼각형이 *덜 덮은* 픽셀

**진짜로 넣을지 (지속성 게이트):**
- 어쩌다 한 번 걸린 곳에 넣으면 노이즈에 GS를 낭비함.
- 그래서 **"같은 카메라(뷰)에서 그 영역이 반복해서 후보로 잡혔는가"** 를 본다.
  (카메라별로 후보 이력을 따로 쌓고, 같은 카메라에서 `min_checkpoint_repeats`회
  이상 반복된 영역만 채택)
- ⚠️ 과거 버전은 *서로 다른 랜덤 뷰*의 픽셀을 비교해서 의미가 없었음 → **카메라별
  비교로 수정함.** (서로 다른 뷰의 같은 픽셀좌표는 전혀 다른 3D 지점이라 비교 불가)

**무엇을 넣나:**
- 채택된 영역에서 잔차 큰 픽셀들을 골라(용량 상한 내에서),
- 그 픽셀을 **삼각형 깊이로 3D 월드좌표에 역투영**한 위치에 GS를 생성.
- 초기값: 색 = 정답 이미지 픽셀, 불투명도 = 0.1, 크기 = 0.01(등방), 회전 = 기본.

## 4. 어떻게 학습하는가 (손실 + 합성)

**합성 방식:** 기본 `depth_aware` — 삼각형과 GS의 **깊이를 비교**해 앞에 있는 걸 위로
얹는 2단 over 합성. (`over`로 바꾸면 깊이 무시하고 GS를 무조건 위에 얹음)

**손실 (매 iteration):**
```
loss = 이미지손실 + 삼각형weight손실 + 법선손실 + GS손실
```
- **이미지 손실** = L1 + SSIM. GS가 켜진 뒤엔 **합성 이미지(삼각형+GS)** 기준으로 계산.
- **법선 손실** = 원본의 metric3d 법선 일치 (그대로 사용).
- **삼각형 weight 손실** = 원본 정규화 항 (초반 구간).
- **GS 손실** = 두 가지 벌점:
  - *sparsity*: GS를 너무 많이/세게 쓰지 못하게 (불투명도 평균에 벌점)
  - *mask*: **채택 영역 "밖"에서 GS가 기여하면 벌점** → GS가 엉뚱한 데 퍼지지 않게

삼각형과 GS는 **각자 옵티마이저로 따로** 업데이트.

## 5. GS → 삼각형 변환 (C)

1. B 체크포인트(`triangles/` + `gaussians.pt`) 로드
2. 각 GS를 **삼각형 패치로 변환** (GS 1개 → 삼각형 2개, 크기 보정 `size_factor`)
3. 변환된 삼각형을 원본 `TriangleModel`에 주입 → **삼각형만으로 finetune**
4. 최종 평가는 **GS 없이 삼각형 렌더만** (재현성 보장)

## 6. 무엇을 보고 판단하나 (측정 / 시각화)

- **학습 중 자동 저장:** `diagnostics.json` — 숫자만
  (PSNR/SSIM, triangle_count, **gs_count**, **gs_contribution_ratio**, 시간)
  ⚠️ 이미지(PNG)는 자동 저장 안 됨.
- **시각화는 별도 스크립트:** `render_gs_assisted.py` 실행 →
  `T_only / G_only / T_plus_G / GS_red_overlay / alpha·depth 맵` PNG 생성.
  - **`GS_red_overlay`** = 삼각형 렌더 위에 GS가 들어간 위치를 빨갛게 표시
    → "GS가 어디 모이나"라는 **이 실험의 핵심 그림.**

## 7. ★ 핵심적으로 볼 부분 (중요한 순서대로)

1. **`gs_contribution_ratio` — 실험의 생명줄.**
   이게 거의 0이면 "GS가 삼각형을 *못* 도왔다"는 뜻이라 가설 검증 자체가 안 됨.
   (스모크에선 1.3e-6 = 사실상 0이었음. 본 실험에서 반드시 추적.)

2. **구조적 "3중 억압" 주의.** 삼각형이 화면을 불투명하게 꽉 덮는 장면에선:
   - 삽입 게이트(alpha<0.35)가 잘 안 걸리고 +
   - depth_aware 합성에서 GS가 삼각형 뒤로 가려지고 +
   - mask/sparsity 벌점이 GS를 누름
   → **GS가 의미를 갖기 구조적으로 어렵다.** 본 런에서도 0이면 의심 순서:
   `init_scale` 키우기 → `--composite-mode over` → 벌점 가중치 낮추기.

3. **GS 분포 위치 = 연구 질문 그 자체.**
   `GS_red_overlay`로 "삼각형이 약한 구간(얇은 구조·경계·고주파)"에 GS가 모이는지 확인.
   단, **2번이 풀려서 기여가 유의미해진 뒤**라야 의미 있음.

4. **변환 충실도 (C).**
   GS가 기여하던 표현이 삼각형 패치로 바뀐 뒤에도 유지되는가?
   (B의 T+G PSNR ≈ C의 삼각형-only PSNR 인가) → 변환에서 다 날아가면 결론을 못 냄.

5. **삽입이 실제로 일어나는지 (수정분 검증).**
   카메라별 재출현으로 고쳤으니 본 런에서 **gs_count가 0이 아니게 증가**하는지 확인.
   (스모크는 짧아서 여전히 0일 수 있음 → 배관 확인은 게이트를 임시로 풀어 강제삽입으로.)

## 8. 알아둘 갭 / 주의

- **`configs/*.yaml`은 현재 학습에서 안 읽힘** — 정책은 CLI 인자 + 코드 기본값으로 정해짐.
  (yaml의 gs_policy는 문서용. 쓰려면 학습 스크립트에 config 로딩을 붙여야 함.)
- 원본 `scene/triangle_model.py`의 `pytorch3d` import는 **안 쓰이는 죽은 코드**라
  서버에서 삭제함 (법선은 metric3d로 별도 처리).
- 상류 baseline(`third_party/triangle-splatting2`)은 git에 안 들고 다님 →
  머신마다 `git clone --recursive`로 다시 받고 CUDA 확장을 빌드 (README 참고).

---

## 부록: 기본 설정값

**저장 스케줄 (`save_iters`):**
30k 런 기준 `3000, 6000, ..., 30000` (10% 간격). 다른 총 iter면 10%~100%를 10%씩.
`gs_start_iter`가 이 지점들에 없으면 진단용으로 추가 저장.

**GS 시작 시점 (`gs_start_iter`) 기본:**
```
max(5000, start_opacity_floor, start_pruning + 1000)
```

**후보 영역 조건:**
- 광도 잔차 상위 10%
- 삼각형 alpha/기여 < 0.35
- 같은 카메라에서 `min_checkpoint_repeats`(기본 2)회 이상 반복

**용량/벌점 기본값:**
- 삽입 1회 최대 5k GS, 전체 최대 100k GS
- GS opacity/개수에 sparsity 벌점
- 채택영역 밖 GS 기여에 mask 벌점

**저장 시 진단 항목:**
T_only / G_only / T_plus_G / GS_red_overlay 렌더(별도 스크립트),
삼각형 depth/alpha/contribution 맵, GS alpha/contribution 맵,
PSNR/SSIM/LPIPS, 시간, triangle 수, GS 수, GS 기여율.

**합격 기준:**
- A와 B/C 코드 경로가 분리돼 있음
- A/B/C 출력이 같은 루트 아래 분리됨
- B는 삽입 후 0이 아닌 GS 수 + GS 기여율 기록
- C 최종이 삼각형 렌더만으로 재현 가능
