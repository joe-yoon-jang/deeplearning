---
title: "신경망"
last_modified_at: 2021-02-15T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망
---


# 신경망

> 신경망은 가중치 매개변수의 적절한 값을 데이터로 자동으로 학습

## 신경망의 예

![](https://t1.daumcdn.net/cfile/tistory/2117013E5928016429)

- 입력층 은닉층 출력층
  - 입력층 0층
  - 은닉층 1층(사람 눈에는 보이지 않는다)
  - 출력층 2층
```
      0 (b + w1x1+ w2x2 <=0)
y =
      1 (b + w1x1+w2x2 > 0)
```
- x1과 x2라는 두 신호를 입력받아  y를 출력하는 퍼셉트론

```
y = h(b + w1x1 + w2x2)
```
- 입력 신호의 충합이 h(x) 라는 함수를 거쳐 변환되어 y로 출력됨을 의미

```
        0 (x<=0)
h(x) = 
        1 (x>0)
```
- 결국 입력 값 x 가 0보다 크면 1을 돌려주고 그렇지 않으면 0을돌려주는 h(x) 함수가 나옴 
- 결과적으로 위 식 3개는 하는일이 동일하다

## 활성화 함수

> 위 h(x) 함수 같이 입력신호 총합을 출력신호로 변환하는 함수를 활성화 함수라 한다(activation function)

- a = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>
  - 가중치가 달린 입력 신호와 편향의 총합 계산하고 이를 a 라한다
- y = h(a)
  - 위 a를 함수 h()에 넣어 y를 출력한다

> 퍼셉트론에서 활성화 함수는 임계값을 경계로 출력이 바뀐다 이를  계단 함수(step function)이라한다

### 시그모이드 함수

h(x) = 1 / (1 + exp(-x))

- exp(-x)는 e<sup>-x</sup>를 뜻하며 자연 상수 2.7182.. 의 값을 갖는 실수
  - 시그모이드 역시 단순 함수로 입력 주면 아웃을줌
    - ex) h(1.0) = 0.731.., h(2.0) = 0.880

```python
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0
```

- 계단함수의 단순한 구현
- 인수x로 넘파이 배일열 받고싶어 아래와 같이 수정

```python
def step_function(x):
  y = x > 0
  return y.astype(np.int)
```
```python
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x # array([-1., 1., 2.])
y = x > 0
y # array([False, True, True], dtype =bool)
y = y.astype(np.int)
y # array([0,1,1])
```
- 넘파이 배열에 부등호 연산을 수행하면 원소 각각 에 bool 배열 생성
- int형 결과를 원하니 astype  으로 자료형 변환

### 시그모이드 구현

```python
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
```
- np.exp(-x)는 epx(-x) 수식에 해당
- 넘파이도 사용 가능

```python
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
# array([0.26894142, 0.73105858, 0.88079708])
```
- 넘파이 브로드캐스트로 수행이 가능하다

#### 시그모이드 계단 함수 차이

![](https://lh3.googleusercontent.com/proxy/y4AN3d14-KAmFcb_i3VjPA9EcSH7N1BHcycqnN5P8sNYIzO1B-77tKikiiUVSqHon10yd0mGEf1RwusYYteb_RbYA7gv_35nYRBoBt90YsAceSIsEZbc1_luhWAosClwB8c-KHAMoJkprcvInH1upTukZk-gcvyyNpXo8q2XEsP2fEHyg4q77psHnHWviho23_xdifTurMVyj1ZcBL-pJhnMkyUKYX1yxvr3V6rxGdT3ZfEtaPPS-EngHI8MK1OWTxVhNkSmFmO17b81p6uhH4IuoYd71tK3rWCxeRd4vTc)

- 가장 큰 차이는 매끄러움
- 공통점
  - 비선형 함수
    - 선형 함수는 f(x) = ax + b 일때 a와b가 상수인 한개의 직선 함수

- 신경망은 비선형 함수를 사용해야만함
  - 선형 함수는 층을 아무리 깊게 해도 은닉층이 없는 네트워크로 똑같이 만들수 있다

### ReLU함수
> 시그모이드 함수를 대신 주로 사용하는 함수
- ReLU : 입력이 0 을 넘으면 그입력을 그대로 출력하고 0 이하면 0을 출력하는 함수
```
        x (x > 0)
h(x) = 
        0 (x <= 0)
```
```python
def relu(x):
  return np.maximum(0,x) # 둘중 큰값 반환
```

## 다차원 배열 계산
> 넘파이의 다차원 배열을 사용한 계산법

### 다차원 배열

> n차원으로 나열하는 배열

```python
import numpy as np
A = np.array([1,2,3,4])
print(A)
# [1 2 3 4]
np.ndim(A)
# 1
A.shape
# (4,)
A.shape[0]
# 4
```