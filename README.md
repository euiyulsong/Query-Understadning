# HotpotQA 검색 오케스트레이션 실험 결과

## 실험 설정

* Dataset: **HotpotQA**
* 샘플 수: **100**
* 비교 방식:

  * `parallel_plan`
  * `plan_execute`
  * `plan_replan`
* 주요 평가 지표:

  * EM
  * F1
  * 평균 search call 수
  * 평균 LLM call 수
  * 평균 latency
  * 평균 replan 수

## 결과 요약

| Method          |        EM |        F1 | Search Calls | LLM Calls | Latency (s) | Replans |
| --------------- | --------: | --------: | -----------: | --------: | ----------: | ------: |
| `parallel_plan` |     0.120 |     0.147 |         2.27 |      2.00 |    **3.58** |    0.00 |
| `plan_execute`  | **0.210** | **0.255** |     **2.18** |      4.12 |        7.46 |    0.00 |
| `plan_replan`   |     0.200 |     0.254 |         2.71 |  **4.63** |        8.97 |    2.65 |

## 핵심 분석

### 1. 정확도는 `plan_execute`가 가장 좋음

`parallel_plan` 대비 `plan_execute`는:

* EM: `0.12 → 0.21`
* F1: `0.147 → 0.255`

로 크게 상승했다.

특히 EM은 약 **75% 상대 개선**이다.

```text
parallel_plan
EM = 0.12

plan_execute
EM = 0.21
```

HotpotQA처럼 앞 단계에서 찾은 entity를 이용해 다음 검색을 해야 하는 **dependency가 있는 multi-hop QA**에서는, 처음부터 모든 검색을 독립적으로 병렬화하는 것보다 **순차 dependency를 명시적으로 실행하는 방식**이 유리하다는 결과다.

### 2. `plan_replan`은 정확도 향상이 거의 없음

`plan_replan`:

```text
EM = 0.200
F1 = 0.254
```

`plan_execute`:

```text
EM = 0.210
F1 = 0.255
```

거의 동일하다.

오히려 EM은 `plan_execute`가 조금 높다.

즉 이번 결과에서는:

> **실행 중 계획을 계속 수정하는 것이, 처음 세운 계획을 그대로 실행하는 것보다 추가적인 정확도 이득을 주지 못했다.**

라고 볼 수 있다.

### 3. 그런데 Replan 비용은 확실히 큼

`plan_execute`와 `plan_replan`을 비교하면:

|              | Plan Execute | Plan Replan |
| ------------ | -----------: | ----------: |
| Search calls |         2.18 |        2.71 |
| LLM calls    |         4.12 |        4.63 |
| Latency      |        7.46s |       8.97s |
| F1           |    **0.255** |       0.254 |

`plan_replan`은 더 많은 검색과 LLM 호출을 쓰면서 성능은 거의 같다.

Latency는:

```text
7.46s → 8.97s
```

약 **20% 증가**했다.

따라서 현재 조건에서는 `plan_replan`이 Pareto 관점에서 `plan_execute`보다 열세다.

### 4. `parallel_plan`은 매우 빠르지만 정확도 손실이 큼

`parallel_plan`의 장점은 명확하다.

```text
Latency
parallel_plan = 3.58s
plan_execute  = 7.46s
plan_replan   = 8.97s
```

`plan_execute` 대비 latency가 절반 이하 수준이다.

하지만:

```text
F1
parallel = 0.147
execute  = 0.255
```

정확도 손실이 상당하다.

재미있는 점은 search call 자체는:

```text
parallel_plan = 2.27
plan_execute  = 2.18
```

거의 같다는 것이다.

따라서 성능 차이는 **검색 횟수**보다 **검색 순서와 dependency 처리 방식**에서 발생했다고 보는 게 더 자연스럽다.

## 해석

이번 결과는 대략 다음 구조를 지지한다.

```text
질문
 │
 ├─ 독립적인 sub-question
 │      ↓
 │   Parallel search
 │
 └─ 앞 결과가 다음 검색에 필요한 multi-hop
        ↓
    Sequential Plan Execute
```

HotpotQA에서는 두 번째 경우가 많기 때문에 단순 `parallel_plan`이 불리하다.

반면 `plan_replan`은 이론적으로는:

```text
Plan
 ↓
Execute
 ↓
Observation
 ↓
Replan
 ↓
Execute
```

를 통해 잘못된 계획을 복구할 수 있지만, 이번 100개에서는 그 복구 효과보다 **추가 tool/LLM 호출 비용**이 더 컸다.

## 현재까지 결론

가장 중요한 결과는:

> **HotpotQA에서는 “planning 자체”보다 dependency-aware execution이 중요했다.**

정리하면:

```text
Accuracy:
Plan Execute ≈ Plan Replan >> Parallel Plan

Latency:
Parallel Plan >> Plan Execute > Plan Replan
```

따라서 현재까지는 **`plan_execute`가 accuracy/cost 균형이 가장 좋다.**

`plan_replan`은 모든 문제에서 기본 전략으로 쓰기보다는, 초기 계획 실패 가능성이 높은 문제에만 선택적으로 쓰는 게 더 적합해 보인다.

최종적으로 `react`까지 나오면 비교가 더 명확해진다:

```text
Parallel Plan
vs
Plan Execute
vs
Plan Replan
vs
ReAct
```

특히 ReAct의 `EM/F1`, search calls, LLM calls가 나오면 **“처음에 전체 plan을 세우는 게 좋은가, observation마다 다음 행동을 결정하는 게 좋은가”**를 직접 비교할 수 있다.

## HotpotQA Comparison — Parallel Decomposition vs ReAct

동일한 평가셋 100개를 사용했으며, `parallel_decomp`와 `ReAct` 모두 정확히 같은 질문·정답 쌍으로 평가했다.

| Method                     |      EM |        F1 | Avg. Latency |
| -------------------------- | ------: | --------: | -----------: |
| **Parallel Decomposition** |     79% |     0.863 |    **2.65s** |
| **ReAct**                  | **81%** | **0.881** |        5.00s |

Paired 결과는 `parallel only correct = 2`, `react only correct = 4`, `both correct = 77`, `both wrong = 17`이었다. 즉 정확도에서는 ReAct가 근소하게 우세했지만, 차이는 매우 작았다. 반면 latency는 Parallel Decomposition이 **2.65초**, ReAct가 **5.00초**로, Parallel 방식이 약 **1.88배 빠른 응답 속도**를 보였다.

특히 Parallel Decomposition만 맞춘 사례도 존재했다. 예를 들어 `Zakk Wylde and Damon Albarn are both what?`에서 Parallel은 정답인 `singer, songwriter, and multi-instrumentalist`를 맞췄지만 ReAct는 `musicians`로 지나치게 일반화했다. 또한 `which is larger, Hunchun or Shijiazhuang?`에서는 Parallel이 `Shijiazhuang`을 맞춘 반면 ReAct는 `Hunchun`으로 오답을 냈다.

결론적으로 **comparison-type multi-hop QA에서는 단순한 upfront decomposition + 병렬 retrieval만으로도 ReAct와 거의 비슷한 정확도를 유지하면서 wall-clock latency를 절반 수준으로 줄일 수 있었다.** 다만 정확도 자체는 이번 실험에서는 ReAct가 F1 기준 약 **+0.018** 높아, Parallel Decomposition의 주요 이점은 정확도 향상보다는 **효율성과 응답 속도**에 있다고 볼 수 있다.

