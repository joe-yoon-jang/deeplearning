---
title: "RegExp"
last_modified_at: 2022-02-27T18:20:02-05:00
categories:
  - Study
tags:
  - javscript
  - RegExp
---
# 31 RegExp

## 31.1 정규 표현식

- 일정한 패턴을 가진 문자열의 집합을 표현하기 위해 사용하는 형식 언어(formal language)
- 패턴 매칭 기능 제공: 특정 패턴과 일치하는 문자열을 검색하거나 추출 또는 치환 기능

## 31.2 정규 표현식의 생성

- RegExp 객체(정규 표현식 객체) 생성
    - 리터럴과 RegExp 생성자 함수 사용

```jsx
// 정규표현식 리터럴
const regexp = /regexp/i;

// pattern 정규 표현식의 패턴, flags: 정규표현식의 플래그 g,i,m,u,y
new RegExp(pattern[, flags])

// 생성자를 통한 동적 생성
const count = (str, char) => (str.match(new RegExp(char,'gi')) ?? []).length;

count('is this all there is?', 'is'); // 3
count('is this all there is?', 'xx'); // 0
```

## 31.3 RegExp 메서드

- String.prototype.replace, String.prototype.search, String.prototype.split 는 32장에서

### 31.3.1 RegExp.prototype.exec

- exec 메서드: 인수로 전달받은 문자열에 대해 정규 표현식의 패턴을 검색하여 매칭 결과를 배열로 반환 없으면  null  반환
- exec 는 g플래그여도 첫 번쨰 매칭 결과만 반환한다

```jsx
const target = 'Is this all there is ?';
const regExp = /is/;
regExp.exec(target);
// -> ["is", index: 5, input: "Is this all there is?", groups: undefiend] 
```

### 31.3.2 RegExp.prototype.test

- 패턴을 검색하여 매칭 결과를 불리언 값으로 반환

```jsx
const target = 'Is this all there is ?';
const regExp = /is/;
regExp.test(target); // true

```

### 31.3.3. String.prototype.match

- 대상 문자열과 인수로 전달받은 정규 표현식과의 매칭 결과를 배열로 반환
- g 플래그 지정 시 모든 매칭 결과 반환

```jsx
const target = 'Is this all there is ?';
const regExp = /is/g;
target.match(regExp);
// -> ["is", "is"] 
```

## 31.4 플래그

- 플래그는 총 6개 있으나 3개만 살펴본다

| 플래그 | 의미 | 설명 |
| --- | --- | --- |
| i | Ignore case | 대소문자를 구별하지 않고 패턴을 검색 |
| g | Global | 대상 문자열 내에서 패턴과 일치하는 모든 문자열을 전역 검색 |
| m | Multi line | 문자열의 행이 바뀌더라도 패턴 검색을 계속한다 |

## 31.5 패턴

- 정규 표현식은 패턴과 플래그로 구성
- 패턴
    - 문자열의 일정한 규칙을 표현하기 위해 사용
    - /로 열고 닫으며 문자열의 따옴표는 생략

### 31.5.1 문자열 검색

```jsx
const target = 'Is this all there is ?';
const regExp = /is/ig;
regExp.match(target); // ["Is", index: 0, ~]
target.match(regExp); // ["Is", "is", "is"]

```

### 31.5.2 임의의 문자열 검색

- . 은 임의의 문자 한 개를 의미
- ... 은 임의의 문자 3개를 의미

```jsx
const target = 'Is this all there is ?';
const regExp = /.../ig;
regExp.match(target); // []
target.match(regExp); // ["Is ", "thi", "s a", "ll ", "the", "re ","is?"]
```

### 31.5.3 반복 검색

- {m,n}은 앞선 패턴이 최소 m번 최대n번 반복되는 문자열을 의미
    - 콤마 뒤에 공백이 있으면 정상 동작하지 않으므로 주의
- “+” 앞선 패턴이 최소 한번 이상 반복되는 문자열
    - “+” 는 {1,} 과 같다
- “?” 앞선 패턴이 최대 한번(0번 포함)이상 반복되는 문자열을 의미
    - “?” 는 {0,1}과 같다

```jsx
const target = 'A AA B BB Aa Bb AAA';
// 'A' 가 최소 1번, 최대 2번 반복 문자 전역 검색
const regExp = /A{1,2}/g;
regExp.match(target); // ["A", "AA", "A", "AA", "A"]

// A가 2번 반복 문자 전역 검색
const regExp = /A{2}/g;
regExp.match(target); // ["AA", "AA"]

// A가 최소 2번 반복 문자 전역 검색
const regExp = /A{2, }/g;
regExp.match(target); // ["AA", "AAA"]
```

```jsx
const target = 'A AA B BB Aa Bb AAA';
// A가 최소 한번 이상 반복되는 문자열을 전역 검색
const regExp = /A+/g;
target.match(regExp); // ['A', 'AA', 'A',' AAA']
```

```jsx
const target = 'color colour';
//colo 다음 u가 최대 한번(0포함) 이상 반복되고 r이 이어지는 문자열
const regExp = /colou?r/g;
target.match(regExp); // ['color', 'colour']
```

### 31.5.4 OR 검색

- “|”은 or의 의미를 갖는다
- /A|B/ 는 A or B
- /A+|B+/g
    - A 또는 B가 한번 이상 반복되는 문자열을 전역 검색(분해되지 않은 단어 검색
- /[AB]+/g
    - A또는 B가 한번 이상 반복되는 문자열을 전역 검색
- /[A-Z]+/g
    - 범위를 지정하려면[]내에 - 를 사용한다 A-Z 대문자 알파벳 검색
- \d
    - 숫자를 의미
- \D
    - 숫자가 아닌 문자
- \w
    - 알파벳 숫자 언더스코어를 의미 [A-Za-z0-9_]와 같다
- \W
    - \w와 반대로 동작한다

```jsx
const target = 'A AA B BB Aa Bb';
// A 또는 B를 전역 검색
const regExp = /A|B/g;
target.match(regExp); // [A, A, A, B, B ,B ,A, B]
// A 또는 B를 전역 검색
const regExp = /A|B/g;
target.match(regExp); // [A, A, A, B, B ,B ,A, B]

const regExp = /[AB]+/g;
target.match(regExp); // [A, AA, B, BB ,A, B]

const regExp = /[AB]+/g;
target.match(regExp); // [A, AA, B, BB ,A, B]

const regExp = /[A-Z]+/g;
target.match(regExp); // [A, AA, B, BB ,A, B]
```

### 31.5.5 NOT 검색

- “[]” 내의 ^은 not의 의미
    - [^0-9]: 숫자를 제외한 문자를 의미

```jsx
const target = 'AA BB 12 Aa Bb';
const regExp = /[^0-9]+/g
target.match(regExp); // -> "AA BB "," Aa Bb"
```

### 31.5.6 시작 위치로 검색

- “[]” 밖의 ^ 문자열의 시작을 의미

```jsx
const target = 'https://naver.com';
//https 로 시작하는지 검사
const regExp = /^https/;
regExp.test(target); // true
```

### 31.5.7 마지막 위치로 검색

- “$” 문자열의 마지막 의미

```jsx
const target = 'https://naver.com';
//com으로 끝나는지 검사
const regExp = /com$/;
regExp.test(target); // true
```

## 31.6 자주 사용하는 정규 표현식

### 31.6.1 특정 단어로 시작하는지 검사

```jsx
const url = 'https://example.com'
// http:// 또는 https:// 로 시작하는지 검사
/https?:\/\//.test(url); // true
/^(http|https):\/\/\\.test(url); // true
```

### 31.6.2 특정 단어로 끝나는지 검사

```jsx
const url = 'https://example.com'
// http:// 또는 https:// 로 시작하는지 검사
/com$/.test(url); // true
```

### 31.6.3 숫자로만 이루어진 문자열인지 검사

```jsx
const target = '12345';
/^\d+$/.test(target); // true
```

### 31.6.4 하나 이상의 공백으로 시작하는지 검사

- “\s” : 여러가지 공백 문자(스페이스 탭 등)을 의미(즉 \t \r \n \v \f와 같은 의미)

```jsx
const target = '  Hi';
/^[\s]_/.test(target); // true
```

### 31.6.5 아이디로 사용 가능한지 검사

- 4~10 자리로 이루어진 알파벳 대소문자 또는 수자

```jsx
const id = 'abc123';
/^[A-Za-z0-9]{4,10}/.test(id); // true
```

### 31.6.6 메일 주소 형식에 맞는지 검사

```jsx
const email = 'joe@gmail.com';
/^[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}$/.test(email);
```

### 31.6.7 핸드폰 번호 형식에 맞는지 검사

```jsx
const cellphone = "010-1234-1234";
/^d{3}-d{3,4}-d{4}$/.test(cellphone);
```

### 31.6.8 특수 문자 포함 여부 검사

```jsx
const target = 'abc#123';
(/[^A-Za-z0-9]/gi).test(target); // true
```