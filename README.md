Udacity의 AB test 과정 학습
- https://www.kaggle.com/code/tammyrotem/ab-tests-with-python/notebook\
- https://classroom.udacity.com/courses/ud257/lessons/9983ee8a-91e6-4c2b-9440-66e00839c98e/concepts/56bfbb88-ab4c-4a98-94be-8f0d1ce50f8a
```python
import math as mt
import numpy as np
import pandas as pd
from scipy.stats import norm
```

## Udacity의 A/B 테스팅 과정
Udacity는 웹사이트 또는 모바일 앱의 잠재적인 개선 사항을 테스트하는 데 사용되는 온라인 실험인 분할 테스트라고도 하는 A/B 테스팅을 위한 훌륭한 무료 과정을 게시했습니다. 이 Python 노트북은 최종 프로젝트의 연습 솔루션입니다.
Udacity의 AB Testing 과정은 Google에서 제공하며 A/B 테스트의 설계 및 분석에 중점을 둡니다. 이 과정에서는 실험을 평가하기 위해 메트릭을 선택하고 특성화하는 방법, 충분한 통계적 검정력을 사용하여 실험을 설계하는 방법, 결과를 분석하고 유효한 결론을 도출하는 방법을 다룹니다.

## 실험 개요 <a class="anchor" id="overview"></a>

**실험 이름:** "무료 평가판" Screener. <br>
온라인 교육을 전문으로 하는 웹사이트인 Udacity에서 진행하며, 전반적인 비즈니스 목표는 학생들의 과정 완료를 극대화하는 것입니다.
### 변경 전 현재 상태 <a class="anchor" id="current"></a>
* 이 실험 당시 Udacity 과정은 현재 과정 개요 페이지에 "무료 평가판 시작" 및 "과정 자료 액세스"의 두 가지 옵션이 있습니다. <br>
* 학생이 "무료 평가판 시작"을 클릭하면 신용 카드 정보를 입력하라는 메시지가 표시되고 유료 버전의 무료 평가판에 등록됩니다. 14일 후에는 먼저 취소하지 않는 한 자동으로 청구됩니다.
* 학생이 "강의 자료 보기"를 클릭하면 무료로 동영상 시청 및 퀴즈를 풀 수 있지만 코칭 지원이나 검증된 수료증을 받지 못하며, 피드백을 위해 최종 프로젝트를 제출하지 않습니다.


### 실험 변경 설명 <a class="anchor" id="description"></a>

* 실험에서 Udacity는 학생이 "무료 평가판 시작"을 클릭하면 코스에 할애할 수 있는 시간을 묻는 변경 사항을 테스트했습니다.
* 학생이 주당 5시간 이상을 표시한 경우, 평소와 같이 체크아웃 절차를 거치게 됩니다. 주당 5시간 미만으로 표시된 경우 Udacity 과정은 일반적으로 성공적인 완료를 위해 더 많은 시간을 투자해야 한다는 메시지가 표시되고 학생이 무료로 과정 자료에 액세스할 수 있음을 시사합니다.
* 이 시점에서 학생은 무료 평가판에 계속 등록하거나 강의 자료에 무료로 액세스할 수 있습니다. [이 스크린샷](https://drive.google.com/file/d/0ByAfiG8HpNUMakVrS0s4cGN2TjQ/view)은 실험의 모습을 보여줍니다.

### 실험 가설 <a class="anchor" id="hypothesis"></a>
가설은 이렇게 하면 미리 학생에 대한 더 명확한 기대치를 설정할 수 있으므로 시간이 충분하지 않아 무료 평가판을 종료하는 좌절한 학생의 수를 줄일 수 있다는 것입니다. 과정. 이 가설이 사실이라면 Udacity는 전체 학생 경험을 개선하고 코스를 완료할 가능성이 있는 학생을 지원하는 코치의 능력을 향상시킬 수 있습니다.

### 실험 세부정보 <a class="anchor" id="details"></a>
전환 단위는 쿠키이지만 학생이 무료 평가판에 등록하면 해당 시점부터 사용자 ID로 추적됩니다. 동일한 사용자 ID로 무료 평가판에 두 번 등록할 수 없습니다. 등록하지 않은 사용자의 경우 코스 개요 페이지를 방문할 때 로그인한 경우에도 실험에서 사용자 ID가 추적되지 않습니다.

## 측정항목 선택 <a class="anchor" id="metric"></a>
 성공적인 실험(또는 최소한 안전한 실험)을 위해서는 두 가지 유형의 측정항목이 필요합니다. 불변(Invariate) 및 평가(evaluation) 메트릭.
불변 메트릭은 "온전성 검사"에 사용됩니다. 즉, 실험(모집단의 일부에 대한 변경 사항을 제시한 방식과 데이터를 수집한 방식)이 본질적으로 잘못되지 않았는지 확인합니다. 기본적으로 이는 실험으로 인해 변경되지 않는(영향을 받지 않는) 측정항목을 선택하고 나중에 이러한 측정항목이 대조군과 실험 그룹 간에 크게 변경되지 않도록 하는 것을 의미합니다.<br>
 반면에 평가 메트릭은 변화가 예상되는 메트릭이며 달성하려는 비즈니스 목표와 관련이 있습니다. 각 메트릭에 대해 $Dmin$을 명시합니다. 이는 비즈니스에 실질적으로 중요한 최소 변경을 표시합니다. 예를 들어, 통계적으로 유의미하더라도 유지율이 2% 미만으로 증가하면 비즈니스에 실용적이지 않습니다.

### 불변 메트릭 - 온전성 검사 <a class="anchor" id="invariate"></a>
| 메트릭 이름 | 공식 | $Dmin$ | 표기 |
|:-:|:-:|:-:|:-:|
| 코스 개요 페이지의 쿠키 수  | # 페이지의 고유한 일일 쿠키 | 쿠키 3000개 | $C_k$ |
| 무료 체험 버튼 클릭수  | 클릭한 #개의 고유한 일일 쿠키 | 240 클릭 | $C_l$ |
| 무료 평가판 버튼 클릭률  | $\frac{C_l}{C_k}$ | 0.01 | $CTP$ ||

### 평가 지표 - 성과 지표 <a class="anchor" id="evaluation"></a>
| 메트릭 이름 | 공식 | $Dmin$ | 표기 |
|:-:|:-:|:-:|:-:|
| 총 전환 | $\frac{enrolled}{C_l}$ | 0.01 | $Conversion_{Gross}$ |
| 보유(유지) | $\frac{paid}{enrolled}$ | 0.01 | $Retention$ |
| 순 전환 | $\frac{paid}{C_l}$ | 0.0075 | $Conversion_{Net}$ |

## 측정항목의 기준 값 추정 <a class="anchor" id="baseline"></a>
실험을 시작하기 전에 이러한 측정항목이 변경되기 전에 어떻게 작동하는지, 즉 기준 값이 무엇인지 알아야 합니다.
### 추정 데이터 수집 <a class="anchor" id="collect"></a>
Udacity는 이러한 측정항목에 대해 다음과 같은 대략적인 추정치를 제공합니다(일일 트래픽에 대한 집계에서 수집된 것으로 추정됨) <br>

| 항목 | 설명 | 추정기 |
|:-:|:-:|:-:|
| 쿠키 수(Cookies) | 코스 개요 페이지를 보기 위한 일일 고유 쿠키 | 40,000 |
| 클릭수(Clicks) | 무료 평가판 버튼을 클릭하는 일일 고유 쿠키 | 3,200 |
| 등록 수(Enrollments) | 일일 무료 평가판 등록 | 660 |
| CTP | CTP 무료 평가판 버튼 | 0.08 |
| 총 전환(GConversion) | 클릭 시 등록 확률 | 0.20625 |
| 보유(Retention) | 등록 시 지불 가능성 | 0.53 |
| 순 전환(NConversion) | 클릭 시 지불 확률 | 0.109313 |


```python
# 추정 데이터를 나중에 사용하기 위해 dictionary로 정의
baseline = {"Cookies":40000,"Clicks":3200,"Enrollments":660,"CTP":0.08,"GConversion":0.20625,
           "Retention":0.53,"NConversion":0.109313}
baseline
```




    {'Cookies': 40000,
     'Clicks': 3200,
     'Enrollments': 660,
     'CTP': 0.08,
     'GConversion': 0.20625,
     'Retention': 0.53,
     'NConversion': 0.109313}



### 표준 편차 추정 <a class="anchor" id="sd"></a>
이러한 추정치를 수집한 후에는 메트릭의 표준 편차를 추정해야 합니다. 이는 샘플 크기 계산 및 결과에 대한 신뢰 구간을 위해 계산됩니다. 메트릭의 변형이 많을수록 중요한 결과에 도달하기가 더 어렵습니다. 하루에 코스 개요 페이지를 방문하는 쿠키의 샘플 크기가 5,000개라고 가정하면(프로젝트 지침에 제공된 대로) - 평가 지표에 대해서만 표준 편차를 추정하려고 합니다. 우리가 고려하고 있는 표본 크기는 우리가 수집한 "모집단(population)"보다 작아야 하고 해당 크기를 가진 두 그룹을 가질 수 있을 만큼 작아야 합니다.

#### 수집된 데이터 스케일링 <a class="anchor" id="scale"></a>
따라야 할 모든 계산을 위해 우리는 분산 추정을 위해 지정한 샘플 크기로 메트릭의 수집된 수 추정치를 스케일링해야 합니다. 이 경우 하루에 코스 개요 페이지를 방문하는 고유 쿠키가 40000개에서 5000개로 변경됩니다.

###### 데이터 스케일링
- 데이터 전처리 과정의 하나
- 데이터의 값이 너무 크거나 작은 경우 모델 알고리즘 학습 과정에서 0으로 수렴하거나 무한으로 발산해버릴 수 있기 때문에 scailing 필요


```python
# 추정치 스케일링
baseline["Cookies"] = 5000  # 4.2.1 40000 -> 5000
baseline["Clicks"]=baseline["Clicks"]*(5000/40000) # 같은 비율로 scailing
baseline["Enrollments"]=baseline["Enrollments"]*(5000/40000)
baseline
```




    {'Cookies': 5000,
     'Clicks': 400.0,
     'Enrollments': 82.5,
     'CTP': 0.08,
     'GConversion': 0.20625,
     'Retention': 0.53,
     'NConversion': 0.109313}



#### 분석적 추정 <a class="anchor" id="estimate"></a>
분산을 분석적으로 추정하기 위해 확률($\hat{p}$)인 측정항목이 이항 분포라고 가정할 수 있으므로 표준 편차에 대해 다음 공식을 사용할 수 있습니다. <br>
            * 이항 분포 :  연속된 n번의 독립적 시행에서 각 시행이 확률 p를 가질 때의 이산 확률 분포

<center><font size="4">$SD=\sqrt{\frac{\hat{p}*(1-\hat{p})}{n}}$</font></center><br>

이 가정은 실험의 **전환 단위(unit of diversion)**가 **분석 단위(unit of analysis)**(메트릭 공식의 분모)와 동일한 경우에만 유효 합니다. 이것이 유효하지 않은 경우 실제 분산은 다를 수 있으므로 경험적으로 추정하는 것이 좋습니다.

각 측정항목에 대해 두 개의 변수를 공식에 연결해야 합니다. <br>
$\hat{p}$ - 발생할 이벤트의 기준 확률 <br>
$ n $ - 샘플 크기 <br>
* **총 전환(Gross Conversion)** - 총 전환의 기준 확률은 __무료 평가판에 등록한 사용자 수__ 를 __무료 평가판을 클릭하는 쿠키 수__ 로 나눈값.($\frac{enrolled}{C_l}$)<br>
 즉, __클릭이 주어졌을 때 가입할 확률__ 입니다. <br>
  이 경우 샘플을 구별하여 대조군과 실험군에 할당하는 요소인 전환 단위(쿠키)는 공식의 분모인 분석 단위(클릭하는 쿠키)와 동일합니다. 총 전환(GC)을 계산합니다. 이 경우 분산에 대한 이 분석적 추정으로 충분합니다.


```python
# 총 전환(GC)에 필요한 p와 n 구하기
# 4자리 십진수로 반올림된 표준 편차(sd)를 계산
GC = {}
GC["d_min"]=0.01   # 3.2 평가지표
GC["p"]=baseline["GConversion"]  # p는 4.1에 주어져 있음 - 없다면 enrollments/clicks로 계산
GC["n"]=baseline["Clicks"]          # 표본크기 : 무료 평가판을 클릭하는 쿠키 수
GC["sd"]=round(mt.sqrt((GC["p"]*(1-GC["p"]))/GC["n"]),4) 
GC["sd"]
```




    0.0202



* **유지(Retention)** - 기본 유지 확률은 유료 사용자 수(무료 14일 후 등록)를 총 등록 사용자 수로 나눈 값.( $\frac{paid}{enrolled}$ )<br>
즉, __등록이 주어졌을 때 지불할 확률__ 입니다. 표본 크기(n)는 등록된 사용자 수입니다.<br>
이 경우 전환 단위는 분석 단위(등록한 사용자)와 같지 않으므로 분석적 추정이 충분하지 않습니다. 이러한 추정에 대한 데이터가 있다면 이 분산도 경험삼아 추정하고 싶을 것입니다.


```python
# Retention(R)에 필요한 p와 n을 구하기
# 4자리 십진수로 반올림된 표준 편차(sd)를 계산
R = {}
R['d_min']=0.01     # 3.2 평가지표
R['p'] = baseline['Retention']
R['n'] = baseline['Enrollments']   # 표본크기 : 등록된 사용자 수
R['sd'] = round( mt.sqrt( ( R['p']*(1-R['p']) ) / R['n'] ), 4 )
R
```




    {'d_min': 0.01, 'p': 0.53, 'n': 82.5, 'sd': 0.0549}



* **순 전환(Net Conversion)** - 순 전환의 기준 확률은 유료 사용자 수를 무료 평가판 버튼을 클릭한 쿠키 수로 나눈 것.($\frac{paid}{C_l}$)<br> 즉, __클릭이 주어졌을 때 지불할 확률__ 입니다. 샘플 크기(n)는 클릭한 쿠키의 수입니다. 이 경우 분석 단위와 전환 단위가 동일하므로 분석적으로 충분히 좋은 추정을 기대합니다. 


```python
# Net Conversion(NC)에 필요한 p와 n을 구하기
# 4자리 십진수로 반올림된 표준 편차(sd)를 계산
NC = {}
NC['d_min']=0.0075
NC['p'] = baseline['NConversion']
NC['n'] = baseline['Clicks']
NC['sd'] = round( mt.sqrt( (NC['p']*(1-NC['p']))/NC['n'] ), 4 )
NC
```




    {'d_min': 0.0075, 'p': 0.109313, 'n': 400.0, 'sd': 0.0156}



##  실험 크기 조정 <a class="anchor" id="sizing"></a>
이 시점에서 기준선(가장 중요하게는 추정된 분산)에서 메트릭을 추정하면 실험이 유의미함과 함께 충분한 통계적 검정력을 갖도록 필요한 샘플 수를 계산할 수 있습니다.

$\alpha=0.05$(유의 수준) 및 $\beta=0.2$(검정력)이 주어지면 실험에 필요한 총 페이지뷰(과정 개요 페이지를 조회한 쿠키) 수를 추정하려고 합니다. 이 양은 대조군과 실험군이라는 두 그룹으로 나뉩니다. 이 계산은 [online calculator](http://www.evanmiller.org/ab-testing/sample-size.html)를 사용하거나 필요한 공식을 사용하여 직접 계산하여 수행할 수 있습니다.

- 제1종 오류: $\alpha$
- 검정력: $1-\beta$
- 검출 가능 효과: $d$
- 기본전환율 $p$의 확률을 제공하는 대조군(cont) 및 실험군(exp)의 최소 표본 크기
    - 귀무가설(simple hypothesis) $H_0 : P_{cont } - P_{exp} = 0$ 
    - 대립가설(against simple alternative) $H_A : P_{cont} - P_{exp} = d$는 다음과 같습니다.

<center> <font size="5"> $n = \frac{(Z_{1-\frac{\alpha}{2}}sd_1 + Z_{1-\beta}sd_2)^2}{d^2 }$</font>, with: <br><br>
$sd_1 = \sqrt{p(1-p)+p(1-p)}$<br><br>
$sd_2 = \sqrt{p(1-p)+(p+d)(1-(p+d))}$
</center><br>

이제 우리가 필요로 하는 입력과 여전히 수행해야 하는 계산을 분석해 보겠습니다. 입력과 관련하여 필요한 모든 데이터가 있습니다.<br>
제 1종 오류($\alpha$), 검정력($1-\beta$), 검출 가능한 변화($d = D_{min}$) 및  기본전환율(Baseline Conversion Rate)(our $\hat{p}$ ).<br>
계산해야 할 사항:
* $1-\frac{\alpha}{2}$ 및 $1-\beta$에 대해 Z score 받기
* 표준편차 1 & 2, 즉 기본과 예상 변화율 모두를 구합니다.<br>
이 모든 구성 요소는 마침내 필요한 수를 산출합니다.

###  z-score 임계값 및 표준 편차 가져오기 <a class="anchor" id="side_methods"></a>
우리는 이 값을 테이블에서 찾는 데 익숙하지만, 정규 분포에 필요한 모든 방법을 얻기 위해 python의 `scipy.stats.norm` 패키지를 사용할 수 있습니다.<br>
`ppf` 방법은 [백분위수 함수 (Percent Point Function_ppf)](https://en.wikipedia.org/wiki/Quantile_function) 또는 Quantile Function에 대한 액세스를 제공하며 [누적 분포 함수 (Cummulative Distribution Function_cdf)](https://en.wikipedia.org/wiki/Cumulative_distribution_function)의 역이 되는 것 외에도 이것은 필요한 임계 **z-score**를 반환하는 함수입니다.


```python
#Inputs: 필수 알파 값(알파는 이미 필요한 테스트에 적합해야 함)
#Returns: 주어진 알파에 대한 z-score
def get_z_score(alpha):
    return norm.ppf(alpha)
```


```python
# 표준편차 1&2, 기본변화율(baseline)과 예상 변화율(expected) 모두 구하기
# Inputs: p(정했던 기본전환율), d(최소 검출가능변화-d_min)
def get_sds(p,d): 
    sd1 = mt.sqrt( 2*p*(1-p) )
    sd2 = mt.sqrt( p*(1-p) + (p+d)*(1-(p+d)) )
    sds = [sd1, sd2]
    return sds
```


```python
# Inputs:sd1,sd2,alpha,beta,d_min,p(기본추정치)
# Returns: 공식에 따라 그룹당 필요한 최소 샘플 크기
def get_sampSize(sds, alpha, beta, d):
    n = pow( (get_z_score(1-alpha/2)*sds[0] + get_z_score(1-beta)*sds[1]),2 ) / pow(d,2)  # pow : 제곱
    return n
```

### 측정항목별 샘플 크기 계산 <a class="anchor" id="calc"></a>
좋아요! 이 부분에 필요한 모든 도구를 설정한 것 같습니다. 이제 메트릭당 실험에 필요한 샘플 수를 계산할 것이며 가장 큰 샘플 크기가 유효 크기가 될 것이라는 사실에 종속됩니다. 이 크기는 기간 및 노출의 효율성 측면에서 고려해야 합니다. 즉, 실험을 위해 이 많은 샘플을 얻는 데 시간이 오래 걸릴 것입니다.

따라서 더 쉽게 작업할 수 있도록 각 메트릭의 각 메트릭 특성에 d 매개변수를 추가해 보겠습니다.


```python
GC, R, NC
```




    ({'d_min': 0.01, 'p': 0.20625, 'n': 400.0, 'sd': 0.0202},
     {'d_min': 0.01, 'p': 0.53, 'n': 82.5, 'sd': 0.0549},
     {'d_min': 0.0075, 'p': 0.109313, 'n': 400.0, 'sd': 0.0156})




```python
GC['d'] = 0.01
R['d'] = 0.01
NC['d'] = 0.0075
GC, R, NC
```




    ({'d_min': 0.01, 'p': 0.20625, 'n': 400.0, 'sd': 0.0202, 'd': 0.01},
     {'d_min': 0.01, 'p': 0.53, 'n': 82.5, 'sd': 0.0549, 'd': 0.01},
     {'d_min': 0.0075, 'p': 0.109313, 'n': 400.0, 'sd': 0.0156, 'd': 0.0075})



Now, for the calculations
#### 총전환(Gross Conversion)


```python
# 정수로 계산
# get_sds(p,d)
# get_sampSize(sds, alpha, beta, d)
GC['SampSize'] = round( get_sampSize(get_sds(GC["p"], GC['d']), 0.05, 0.2, GC['d'] ) )
GC['SampSize']
```




    25835



즉, 무료 평가판 버튼을 클릭하는 그룹당 최소 25,835명의 쿠키가 필요합니다!<br> 
즉, 5000 페이지뷰 중 400 클릭이 발생했다면 (`400/5000 = 0.08`($CTP$)) ->  그룹당 `GC["SampSize"]/0.08 = 322,938` 페이지뷰가 필요합니다.<br>
마지막으로, 총 전환 측정항목당 샘플의 총량은 다음과 같습니다.


```python
GC['SampSize']=round(GC["SampSize"]/0.08*2)   # 그룹당 GC["SampSize"]/0.08 -> *2
GC['SampSize']
```




    645875



#### 유지(Retention)


```python
R['SampSize'] = round( get_sampSize( get_sds(R['p'], R['d']), 0.05, 0.2, R['d'] ) )
R['SampSize']
```




    39087



이는 그룹당 39,087명의 사용자가 등록해야 함을 의미<br>
먼저 이것을 클릭한 쿠키로 변환한 다음 페이지를 본 쿠키로 변환해야 합니다. 마지막으로 두 그룹에 대해 2를 곱해야 한다 


```python
R['SampSize'] = R['SampSize']/0.08/0.20625*2   # 0.20625 : 총 전환(enrolled/clicks)
R['SampSize']
```




    4737818.181818182



이것은 총 400만 이상의 페이지 뷰를 필요로 합니다. 이것은 우리가 하루에 약 40,000개를 얻는다는 것을 알고 있기 때문에 사실상 불가능합니다.<br>
이것은 100일이 훨씬 더 걸릴 것입니다. 이것은 우리가 실험(훨씬 더 작음)의 결과가 편향될 것이기 때문에 이 측정항목을 삭제하고 계속해서 작업하지 않아야 함을 의미합니다.

#### 순전환(Net Conversion)


```python
NC['SampSize'] = round( get_sampSize( get_sds(NC['p'],NC['d']), 0.05, 0.2, NC['d'] ) )
NC['SampSize']
```




    27413



따라서 그룹당 클릭하는 쿠키 27,413개가 필요하면 다음과 같은 결과를 얻을 수 있습니다.


```python
NC['SampSize'] = NC['SampSize']/0.08*2
NC['SampSize']
```




    685325.0



우리는 페이지를 보는 최대 685,325명의 쿠키입니다. <br>
이것은 총 전환(Gross Conversion)에 필요한 것보다 많으므로 이것이 우리의 수치가 될 것입니다.<br>
매일 페이지뷰의 80%를 차지한다고 가정하면 이 실험의 데이터 수집 기간(실험이 공개되는 기간)은 약 3주입니다.

## 수집된 데이터 분석 <a class="anchor" id="analysis"></a>
마침내, 우리 모두가 기다려온 그 순간, 우리는 마침내 이 실험이 증명할 것을 보게 됩니다!
데이터는 두 개의 스프레드시트로 표시됩니다. 각 스프레드샷을 pandas 데이터 프레임에 로드합니다.

### 수집된 데이터 로드 <a class="anchor" id="collect_results"></a>


```python
control = pd.read_csv('./data/control_data.csv')      # 대조군
experiment=pd.read_csv("./data/experiment_data.csv")  # 실험군
control.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10511</td>
      <td>909</td>
      <td>167.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9871</td>
      <td>836</td>
      <td>156.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>10014</td>
      <td>837</td>
      <td>163.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



### 온전성 검사(Sanity Checks) <a class="anchor" id="sanity"></a>
이 실험의 결과를 분석하기 전에 먼저 해야 할 일은 온전성 검사(Sanity Checks)입니다. 이러한 확인은 실험이 예상대로 수행되었고 다른 요소가 수집한 데이터에 영향을 미치지 않았는지 확인하는 데 도움이 됩니다. 이것은 또한 데이터 수집이 올바른지 확인합니다.

다음과 같은 3가지 고정 측정항목이 있습니다.
* 코스 개요 페이지의 쿠키 수($C_k$)
* 무료 평가판 버튼 클릭 수($C_l$)
* 무료 평가판 버튼 클릭률($CTP$)

이러한 측정항목 중 두 가지는 쿠키 수 또는 클릭수와 같은 단순 계수이고 세 번째는 확률($CTP$)입니다.<br>
이러한 관찰된 값이 예상과 같은지 확인하는 두 가지 다른 방법을 사용할 것입니다(실제로 실험이 손상되지 않은 경우).

####  개수간의 차이에 대한 온전성 검사 <a class="anchor" id="check_counts"></a> 
* **과정 개요 페이지를 본 쿠키 수($C_k$)** - 이 단순 불변 측정항목에서 시작하여 각 그룹으로 전환한 쿠키 페이지뷰의 총량을 계산하고 쿠키 양에 상당한 차이가 있는지 확인하려고 합니다. 상당한 차이는 결과에 의존해서는 안 되는 편향된 실험을 의미합니다.


```python
pageviews_cont = control['Pageviews'].sum()    # 대조군의 pageviews 총합
pageviews_exp = experiment['Pageviews'].sum()  # 실험군의 pageviews 총합
pageviews_total = pageviews_cont + pageviews_exp
print("number of pageviews in control:", pageviews_cont)
print("number of pageviews in experiment:", pageviews_exp)
```

    number of pageviews in control: 345543
    number of pageviews in experiment: 344660


Good, 이것은 꽤 가까운 숫자처럼 보입니다. 이제 이 양의 차이가 중요하지 않고 무작위이며 우리가 예상한 것과 같은지 확인하겠습니다. 이 전환을 다음과 같은 방식으로 모델링할 수 있습니다. <br>
대조군의 페이지뷰 수는 두 그룹의 전체 페이지뷰의 약 절반(50%)이 될 것으로 예상하므로 사용하기 쉬운 분포로 임의 변수를 정의할 수 있습니다. <br>
이항 확률 변수는 단일 성공 확률이 주어지면 N번의 실험에서 나올 것으로 기대할 수 있는 성공 횟수입니다. 따라서 확률이 0.5인 성공 그룹(예: 통제)에 할당되는 것을 고려한다면 그룹에 할당되는 샘플의 수는 무작위 이항 변수의 값입니다! <br>

이것은 평균이 $p$이고 표준편차가 $\sqrt{\frac{p(1-p)}{N}}$인 경우를 사용하여 이항 분포를 정규 분포(n이 충분히 클 때)에 근사하도록 하는 중심 극한 정리 덕분에 더 쉽습니다. 
<center> <font size="4"> $ X$~$N( p,\sqrt{\frac{p(1-p)}{N}})$ </font></center>
우리가 테스트하고자 하는 것은 관찰된 $\hat{p}$(대조군의 샘플 수를 두 그룹의 총 샘플 수로 나눈 값)이 $p=0.5$와 크게 다르지 않은지 여부입니다.<br>
이를 위해 95% 신뢰 수준에서 허용 가능한 오차 한계를 계산할 수 있습니다.
<center> <font size="4"> $ ME=Z_{1-\frac{\alpha}{2}}SD$ </font></center>

마지막으로, [신뢰 구간($CI$)](https://en.wikipedia.org/wiki/Confidence_interval)은 관찰된 $p$가 존재할 수 있고 예상 값과 "동일한" 것으로 받아들일 수 있는 범위를 알려주기 위해 파생될 수 있습니다.
<center> <font size="4"> $ CI=[\hat{p}-ME,\hat{p}+ME]$ </font></center>


관찰된 $\hat{p}$가 이 범위 내에 있으면 모든 것이 정상이고 테스트가 통과되었습니다. -> **양의 차이는 중요하지 않고 무작위이며 예상 값과 동일하다.**


```python
p = 0.5
alpha = 0.05
p_hat = round(pageviews_cont/pageviews_total, 4)
sd = mt.sqrt( p*(1-p)/(pageviews_total) )
ME = round( get_z_score(1-(alpha/2))*sd, 4 )
print(f'신뢰구간은 {p-ME} 과 {p+ME} 사이에 있습니다. {p_hat}은 신뢰구간 내에 있나요?')
```

    신뢰구간은 0.4988 과 0.5012 사이에 있습니다. 0.5006은 신뢰구간 내에 있나요?


우리가 관찰한 $\hat{p}$는 이 범위 안에 있으며, 이는 그룹 간의 샘플 수의 차이가 예상된다는 것을 의미합니다. 이 __불변 메트릭(invariant metric) 온전성 테스트(sanity test)__ 를 통과했기 때문에 지금까지는 좋습니다!
* **무료 체험 버튼을 클릭한 쿠키 수($C_l$)**
우리는 이전과 동일한 전략으로 이 수를 다룰 것입니다.


```python
clicks_cont = control['Clicks'].sum()
clicks_exp = experiment['Clicks'].sum()
clicks_total = clicks_cont+clicks_exp
```


```python
p_hat = round(clicks_cont/clicks_total, 4)
sd = mt.sqrt( p*(1-p)/clicks_total )
ME = round(get_z_score(1-(alpha/2))*sd, 4)
print(f'신뢰구간은 {p-ME} 과 {p+ME} 사이에 있습니다. {p_hat}은 신뢰구간 내에 있나요?')
```

    신뢰구간은 0.4959 과 0.5041 사이에 있습니다. 0.5005은 신뢰구간 내에 있나요?


우리에게는 또 다른 패스가 있습니다! 좋습니다. 지금까지는 실험 결과로 모든 것이 잘 된 것 같습니다. 이제 확률인 최종 메트릭입니다.

#### 확률 간의 차이에 대한 온전성 검사<a class="anchor" id="check_prob"></a> 
* __무료 평가판 버튼의 클릭률($CTP$)__ <br>

이 경우 무료평가판 버튼 클릭율($CTP$($\frac{C_l}{C_k}$))이 두 그룹에서 거의 동일한지 확인(이는 실험으로 인해 변경될 것으로 예상되지 않았기 때문).<br> 이를 확인하기 위해 각 그룹의 $CTP$를 계산하고 이들 사이의 예상 차이에 대한 신뢰 구간을 계산합니다.

즉, 계산된 신뢰 구간에 따라 허용 가능한 오차 범위와 함께 차이가 없을 것으로 예상합니다($CTP_{exp}-CTP_{cont}=0$).<br>
우리가 주목해야 할 변경 사항은 표준 오차의 계산을 위한 것입니다. 이 경우에는 합동 표준 오차입니다.
<br>

<center><font size="4">$SD_{pool}=\sqrt{\hat{p_{pool}}(1-\hat{p_{pool}}(\frac{1}{N_{cont} }+\frac{1}{N_{exp}})}$</font></center>
<br> with <center><font size="5"> $\hat{p_{pool}}=\frac{x_{cont}+x_{exp}}{N_{cont}+N_{exp}}$ </font></center>
<br><center><font size="1"> $(x:클릭, N:페이지 뷰)$ </font></center>


$CTP$는 페이지뷰 수 중 클릭 수와 같은 __모집단에서의 비율__ (모집단 n의 이벤트 x 수)이라는 점을 이해해야 합니다.


```python
ctp_cont = clicks_cont/pageviews_cont    # 대조군 클릭율
ctp_exp = clicks_exp/pageviews_exp       # 실험군 클릭율

d_hat = round(ctp_exp-ctp_cont, 4)       # 실험군 클릭율과 대조군 클릭율의 차이
p_pooled = clicks_total/pageviews_total  # pooled: 합동 데이터(실험군+대조군)
sd_pooled = mt.sqrt( p_pooled*(1-p_pooled)*(1/pageviews_cont+1/pageviews_exp)) 

ME = round( get_z_score(1-(alpha/2))*sd_pooled, 4)
print(f'신뢰구간은 {0-ME}와 {0+ME} 사이입니다. {d_hat}은 신뢰구간 내에 있나요?')
```

    신뢰구간은 -0.0013와 0.0013 사이입니다. 0.0001은 신뢰구간 내에 있나요?


Wonderful! 이번 테스트도 잘 통과한 것 같습니다.

### 효과 크기 검사 <a class="anchor" id="effect"></a>
다음 단계는 우리의 평가 지표와 관련하여 대조군과 실험군 사이의 변화를 살펴보고 차이가 있는지, 통계적으로 유의미하고 가장 중요하게는 실질적으로 유의미한지 확인합니다(차이가 회사에 유익한 변화를 주는 실험 대상이 될 만큼 충분히 "크다").

이제 남은 것은 각 평가 메트릭에 대해 두 그룹의 값 간의 차이를 측정하는 것입니다. 그런 다음 차이에 대한 신뢰 구간을 계산하고 이 신뢰 구간이 통계적으로나 실질적으로 유의한지 여부를 테스트합니다.

#### 총 전환(Gross Conversion)
메트릭은 신뢰 구간에 __0이 포함되지 않은 경우__(즉, 변경이 있다고 확신할 수 있는 경우)와 신뢰 구간에 __실질적인 유의성 경계($D_min$)가 포함되지 않은 경우__ 통계적으로 유의하다.(즉, 비즈니스에 중요한 변화가 있다고 확신합니다.) 

> **중요:** 주어진 스프레드시트에는 39일 동안의 페이지뷰 및 클릭수가 나열되지만 23일 동안의 등록 및 지불만 나열됩니다. 따라서 등록 및 지불 작업을 할 때 모든 페이지뷰와 클릭이 아니라 해당하는 페이지뷰와 클릭만 사용한다는 사실을 알아야 합니다.


```python
# 총 클릭수만 계산
clicks_cont = control['Clicks'].loc[control["Enrollments"].notnull()].sum() 
clicks_exp = experiment['Clicks'].loc[experiment["Enrollments"].notnull()].sum()
clicks_cont, clicks_exp
```




    (17293, 17260)




```python
#Gross Conversion(총전환) - number of enrollments divided by number of clicks(등록/클릭)
enrollments_cont = control['Enrollments'].sum()
enrollments_exp = experiment['Enrollments'].sum()

GC_cont = enrollments_cont/clicks_cont
GC_exp = enrollments_exp/clicks_exp

GC_pooled = (enrollments_cont+enrollments_exp)/(clicks_cont+clicks_exp)
GC_sd_pooled = mt.sqrt( GC_pooled*(1-GC_pooled)*(1/clicks_cont+1/clicks_exp) )
GC_ME = round( get_z_score(1-alpha/2)*GC_sd_pooled, 4 )
GC_diff = round(GC_exp - GC_cont, 4)

print(f"실험으로 인한 변화는 {GC_diff*100}% 입니다.")
print(f'신뢰구간(CI) : [{GC_diff-GC_ME} , {GC_diff+GC_ME}]')
print(f'CI에 0이 포함되어 있지 않으면 변화가 통계적으로 유의미합니다. 이 경우 {-GC["d_min"]}이 CI에 포함되어있지 않다면 실질적으로 유의미합니다.')
```

    실험으로 인한 변화는 -2.06% 입니다.
    신뢰구간(CI) : [-0.0292 , -0.012]
    CI에 0이 포함되어 있지 않으면 변화가 통계적으로 유의미합니다. 이 경우 -0.01이 CI에 포함되어있지 않다면 실질적으로 유의미합니다.


이 결과에 따르면 실험으로 인한 변화가 있었고 그 변화는 통계적으로나 실질적으로 유의미했습니다.
1%보다 큰 변화를 기꺼이 받아들일 때 2.06%의 음수 변화가 있습니다. 즉, 실험군(변화에 노출된 그룹, 즉 공부에 몇 시간을 할애할 수 있는지 묻는 그룹)의 총 전환율($Conversion_{Gross}$)이 예상대로 2% 감소했으며 이러한 변화가 상당했습니다. 즉, 팝업으로 인해 무료 평가판에 등록한 사람이 줄어듭니다.

#### 순 전환율(Net Conversion)
가설은 총 전환 대신 순 전환만 이전과 동일합니다. 이 시점에서 지불자 비율(클릭 외)도 감소할 것으로 예상합니다.


```python
#Net Conversion - number of payments divided by number of clicks
payments_cont = control['Payments'].sum()
payments_exp = experiment['Payments'].sum()

NC_cont = payments_cont/clicks_cont
NC_exp = payments_exp/clicks_exp

NC_pooled = (payments_cont+payments_exp)/(clicks_cont+clicks_exp)
NC_sd_pooled = mt.sqrt(NC_pooled*(1-NC_pooled)*(1/clicks_cont+1/clicks_exp))
NC_ME = round( get_z_score(1-alpha/2)*NC_sd_pooled, 4 )
NC_diff = round( NC_exp-NC_cont, 4 )

print(f'실험으로 인한 변화는 {NC_diff*100}% 입니다')
print(f'신뢰구간(CI): {[NC_diff-NC_ME, NC_diff+NC_ME]}')
print(f'CI에 0이 포함되지 않은 경우 변화가 통계적으로 유의합니다. 이 경우 {NC["d_min"]}가 CI에도 포함되지 않으면 실질적으로 유의합니다. ')
```

    실험으로 인한 변화는 -0.49% 입니다
    신뢰구간(CI): [-0.0116, 0.0018000000000000004]
    CI에 0이 포함되지 않은 경우 변화가 통계적으로 유의합니다. 이 경우 0.0075가 CI에도 포함되지 않으면 실질적으로 유의합니다. 


이 경우에 우리는 0.5% 미만의 변화 크기를 얻었습니다. 통계적으로 유의하지 않은 매우 작은 감소이므로 실질적으로 유의하지 않습니다.

### Sign Test로 더블체크 <a class="anchor" id="sign_tests"></a>
Sign Test에서 우리는 우리가 얻은 결과를 분석할 때 또 다른 각도를 얻습니다. 우리는 관찰한 변화(증가 또는 감소)의 추세가 일일 데이터에서 분명한지 확인합니다.<br> 우리는 하루 당 메트릭 값을 계산한 다음 실험 그룹에서 메트릭이 얼마나 낮았는지 계산할 것이고 이것이 이항 변수에 대한 성공 횟수가 될 것입니다. 이것이 정의되면 사용 가능한 모든 날짜 중 성공 날짜의 비율을 볼 수 있습니다.

#### 데이터 준비 <a class="anchor" id="prep"></a>


```python
control.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
experiment.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9288</td>
      <td>785</td>
      <td>116.0</td>
      <td>91.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 두 dataset을 병합
full = control.join(other=experiment, how='inner', lsuffix='_cont', rsuffix='_exp')
full.count()
```




    Date_cont           37
    Pageviews_cont      37
    Clicks_cont         37
    Enrollments_cont    23
    Payments_cont       23
    Date_exp            37
    Pageviews_exp       37
    Clicks_exp          37
    Enrollments_exp     23
    Payments_exp        23
    dtype: int64




```python
# 완전한 데이터만 남기기
full = full.loc[full['Enrollments_cont'].notnull()]
full.count()
```




    Date_cont           23
    Pageviews_cont      23
    Clicks_cont         23
    Enrollments_cont    23
    Payments_cont       23
    Date_exp            23
    Pageviews_exp       23
    Clicks_exp          23
    Enrollments_exp     23
    Payments_exp        23
    dtype: int64




```python
full.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_cont</th>
      <th>Pageviews_cont</th>
      <th>Clicks_cont</th>
      <th>Enrollments_cont</th>
      <th>Payments_cont</th>
      <th>Date_exp</th>
      <th>Pageviews_exp</th>
      <th>Clicks_exp</th>
      <th>Enrollments_exp</th>
      <th>Payments_exp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
      <td>Sat, Oct 11</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
      <td>Sun, Oct 12</td>
      <td>9288</td>
      <td>785</td>
      <td>116.0</td>
      <td>91.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 각각의 metric에 대한 새로운 컬럼을 얻어 데일리 값을 추출
# 실험군의 value가 대조군의 value보다 크다면 1이 필요
x = full['Enrollments_cont']/full['Clicks_cont']  # 대조군 GC
y = full['Enrollments_exp']/full['Clicks_exp']    # 실험군 GC
full['GC'] = np.where(x<y, 1, 0)  # 대조군 < 실험군 -> 1

z = full['Payments_cont']/full['Clicks_cont']     # 대조군 NC
w = full['Payments_exp']/full['Clicks_exp']       # 실험군 NC
full['NC'] = np.where(z<w, 1, 0)  # 대조군 < 실험군 -> 1
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_cont</th>
      <th>Pageviews_cont</th>
      <th>Clicks_cont</th>
      <th>Enrollments_cont</th>
      <th>Payments_cont</th>
      <th>Date_exp</th>
      <th>Pageviews_exp</th>
      <th>Clicks_exp</th>
      <th>Enrollments_exp</th>
      <th>Payments_exp</th>
      <th>GC</th>
      <th>NC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
      <td>Sat, Oct 11</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
      <td>Sun, Oct 12</td>
      <td>9288</td>
      <td>785</td>
      <td>116.0</td>
      <td>91.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10511</td>
      <td>909</td>
      <td>167.0</td>
      <td>95.0</td>
      <td>Mon, Oct 13</td>
      <td>10480</td>
      <td>884</td>
      <td>145.0</td>
      <td>79.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9871</td>
      <td>836</td>
      <td>156.0</td>
      <td>105.0</td>
      <td>Tue, Oct 14</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
      <td>92.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>10014</td>
      <td>837</td>
      <td>163.0</td>
      <td>64.0</td>
      <td>Wed, Oct 15</td>
      <td>9793</td>
      <td>832</td>
      <td>140.0</td>
      <td>94.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
GC_x = full.GC[full["GC"]==1].count()   # 실험군의 value가 대조군보다 큰 경우
NC_x = full.NC[full["NC"]==1].count()
n = full.NC.count()

print("GC 경우의 수:", GC_x)
print("NC 경우의 수:", NC_x)
print("총 경우의 수:", n)
```

    GC 경우의 수: 4
    NC 경우의 수: 10
    총 경우의 수: 23


### 서명 테스트 빌드 <a class="anchor" id="sign"></a>
우리는 이 부분을 모두 잊고 [온라인 사인 테스트 계산기](https://www.graphpad.com/quickcalcs/binomial2/)를 사용할 수 있지만 저에게는 재미가 없습니다. 그래서 뒤에 있는 계산을 구현하겠습니다. 그것. <br>
실험군이 통제군보다 더 높은 메트릭 값을 가졌던 날의 양을 계산한 후 우리가 하고자 하는 것은 그 숫자가 새로운 실험에서 다시 나타날 가능성이 있는지 확인하는 것입니다(통계적 유의성). <br>
우리는 이와 같은 날의 확률이 무작위라고 가정하고(발생할 확률이 50%) $p=0.5$인 이항 분포와 실험 횟수(일)를 사용하여 무작위에 따라 이러한 일이 발생할 확률을 알려줍니다.<br>

따라서 $p=0.5$ 및 $n=$총 일수 인 이항 분포에 따르면; 이제 $x$일이 성공할 확률을 원합니다(실험군에서 더 높은 메트릭 값). 우리는 양측검정(two-tailed test)을 하고 있기 때문에 이 확률을 두 배로 늘리고 싶고 일단 갖게 되면 그것을 $p-value$라고 부르고 $\alpha$와 비교할 수 있습니다.<br> $p-value$가 $\alpha$보다 크면 결과가 중요하지 않으며 그 반대의 경우도 마찬가지입니다.<br>

<center><font size="4"> $p(successes )=\frac{n!}{x!(nx)!}p^x(1-p)^{nx}$ </font></center>
$p-value$는 검정 통계량이 관찰된 것보다 더 극단적으로 관찰될 확률입니다. 그렇게 2일을 관찰했다면 테스트에 대한 $p-value$는 $p-value = P(x <= 2)$입니다. 다음 사항만 기억하면 됩니다.<br>
<center>$P(x<=2)=P(0)+P(1)+P(2)$.</center><br>



자세한 내용은 [이 우수한 페이지](http://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_nonparametric/BS704_Nonparametric5.html).


```python
# 1. x의 확률을 계산하는 함수 = 성공횟수 
def get_prod(x, n):
    p = round(mt.factorial(n)/(mt.factorial(x)*mt.factorial(n-x))*0.5**x*0.5**(n-x), 4)
    return p

# 2. 최대 x의 확률에서 p-value 계산하는 함수
def get_2side_pvalue(x, n):
    p =0
    for i in range(0, x+1):
        p = p + get_prod(i, n)
    
    return 2*p
```

마지막으로 부호 테스트 자체를 수행하기 위해 `GC_x`, `NC_x` 및 `n` 카운트와 `get_2side_pvalue` 함수를 사용하여 각 메트릭에 대한 $p-value$을 계산합니다.


```python
print(f'{get_2side_pvalue(GC_x, n)}이 0.05보다 작으면 GC 변화가 중요합니다.')
print(f'{get_2side_pvalue(NC_x, n)}이 0.05보다 작으면 NC 변화가 중요합니다.')
```

    0.0026000000000000003이 0.05보다 작으면 GC 변화가 중요합니다.
    0.6774이 0.05보다 작으면 NC 변화가 중요합니다.


우리는 효과 크기 계산에서 얻은 것과 동일한 결론을 얻습니다. 즉, 총 변환(GC)의 변화는 실제로 상당히 중요한반면 순 변환(NC)의 변화는 그렇지 않았습니다.

## 결론 및 권장 사항 <a class="anchor" id="conclusions"></a>
이 시점에서 우리가 달성한 실제 기본 목표에 도달하지 않았음을 확인했다면(과정에 투자할 시간이 있는지 미리 물어 유료 사용자의 비율을 높임), 변경을 계속하지 않는 것이 좋습니다. 총 전환율이 변경되었을 수 있지만 순 전환율에는 영향을 미치지 않습니다.

놀라운 Andrew Bauman의 이 실험, 분석 및 결과에 대한 멋진 요약은 [여기](https://github.com/baumanab/udacity_ABTesting#summary)에서 찾을 수 있습니다.


```python

```


```python

```


```python

```


```python

```


```python

```
