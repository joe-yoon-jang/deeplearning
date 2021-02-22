---
title: "오차역전파법"
last_modified_at: 2021-02-22T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망학습
---


# 오차역전파법

> 수치비분은 구현하기 쉽지만 계산시간이 오래걸린다는 단점이 있다 이에 효율적인 오차역전파법을 공부한다

## 계산 그래프

- 100원 짜리 사과2개를 샀다. 지불 금액을 구하라 단 소비세 10%가 부과된다.
  1. 사과100 -> 곱하기 -> 곱하기 -> 지불금액
  2. 사과의 개수 2
  3. 소비세 1.1
  - 사과 개수와 곱해서 200, 이후 소비세와 곱하고 220


- 100원 짜리 사과2개, 150원짜리 귤 3개를 샀다. 지불 금액을 구하라 단 소비세 10%가 부과된다.

```
사과의 개수 2 ->
                  X 200
사과 가격 100 -> 
                          + 650  
                                   X 715
귤 가격 150   -> 
                  X 450
귤의 개수 3   ->
소비세 1.1                      ->
```
1. 계산 그래프를 구성
2. 왼쪽에서 오른쪽으로 계산을 진행(순전파)
  - 반대는 역전파

### 국소적 계산

- 계산을 국소적으로 간단하게 진행한다

### 덧셈 노드 역전파

- z = x + y 라는 미분은 ∂z/∂Lx = 1 ∂z/∂Ly = 1로 해석적으로 계산가능
  - 덧셈 노드 역전파: 입력 값을 그대로 흘려보낸다
  - 10 + 5 = 15 이면 역전파는 1.3 + 1.3 = 1.3 으로 그대로 보낸다
    - 입력 신호를 그대로 출력할뿐이므로 다음 노드로 전달한다

### 곱셈 노드 역전파
- z = xy 의 미분은  ∂z/∂Lx = y  ∂z/∂Ly = x
- 10 * 5 = 50 은 1.3 = 6.5 * 13 으로 역전파 계산이된다

### 앞 문제 역전파 

```
사과의 개수 2(110) ->
                  X 200(1.1)  
사과 가격 100(2.2) -> 
                          + 650(1.1)  
                                   X 715 (1)
귤 가격 150(3.3)   -> 
                  X 450(1.1)  
귤의 개수 3(165)   ->
소비세 1.1                   (650)->
```

## 계층 구현

- forward와 backward 공통 인터페이스로 진행

```python
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
  def backward(self, dout):
    dx = dout * self.y #x와 y를 바꾼다
    dy = dout * self.x
    return dx, dy

class AddLayer:
  def __init__(self):
    pass

  def forward(self, x, y):
    out = x + y
    return out
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy
```
## 활성화 함수 계층 구현학기

> 계산 그래프를 신경망에 적용, ReLU, Sigmoid 계층 구현

### ReLU 계층

```   
      x  (x>0)
y = { 
      0  (x <= 0)


            1 (x > 0)
∂y/∂x   = { 
            0 (x <= 0)
```
* x > 0

x              y
→              →
       relu
←              ←
∂L/∂y        ∂L/∂y

* x <= 0

x              y
→              →
       relu
←              ←
0             ∂L/∂y


```python
class Relu:
  def __init__(self):
    self.mask = None

  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward(self, dout):
    dout[self.mask] = 0
    dx = douit
    return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# array([[ 1. , -0.5], [-2. ,  3. ]])
mask = (x <= 0)
mask
# array([[False,  True], [ True, False]])
```
- Relu 클래스는 mask 라는 인스턴스 변수를 갖는다
  - mask: True, False 로 구성된 넘파이 배열
    - 순전파 입력 X의 원소의 값이 0이하인 인덱스는 True, 그 외는 False

> ReLU 계층은 스위치와 같다. 순전파 떄 전류가 흐르면 ON 아니면OFF로한다. 역전파 떄는 스위치가 ON이라면 전류가 그대로 흐르고 OFF이면 흐르지 않는다.

### sigmoid

