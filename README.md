# machine learning term project 

### Dependencies
```
python 3.6.4
pytorch 1.0.1 post2
```

### 해야할것 
1. 그래프 그리기(train loss function, acc, test acc 등등)  
2. 결과분석  
3. 여러 모델에 대한 실험결과(망 깊이 등등..?)  
4. 추가적인 attack 방법?(optinal)

### Usage
1. running examples
```
sh run.sh 
```
<br>

### Results
#### Non-targeted attack
from the left, legitimate examples, perturbed examples, and indication of perturbed images that changed predictions of the classifier, respectively
1. non-targeted attack, iteration : 1, epsilon : 0.03
![non-targeted1](misc/nontargeted_1.PNG)
2. non-targeted attack, iteration : 5, epsilon : 0.03
![non-targeted2](misc/nontargeted_2.PNG)
1. non-targeted attack, iteration : 1, epsilon : 0.5
![non-targeted3](misc/nontargeted_3.PNG)
<br>

#### Targeted attack
from the left, legitimate examples, perturbed examples, and indication of perturbed images that led the classifier to predict an input as the target, respectively
1. targeted attack(9), iteration : 1, epsilon : 0.03
![targeted1](misc/targetd_9_1.PNG)
2. targeted attack(9), iteration : 5, epsilon : 0.03
![targeted2](misc/targetd_9_2.PNG)
1. targeted attack(9), iteration : 1, epsilon : 0.5
![targeted3](misc/targetd_9_3.PNG)
<br>

