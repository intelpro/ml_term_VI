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
1. train a simple MNIST classifier
```
python main.py --mode train --env_name [NAME]
```
2. load trained classifier, generate adversarial examples, and then see outputs in the output directory
```
python main.py --mode generate --iteration 1 --epsilon 0.03 --env_name [NAME] --load_ckpt best_acc.tar
```
3. for a targeted attack, indicate target class number using ```--target``` argument(default is -1 for a non-targeted attack)
```
python main.py --mode generate --iteration 1 --epsilon 0.03 --target 3 --env_name [NAME] --load_ckpt best_acc.tar
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

### References
1. explaining and harnessing adversarial examples, Goodfellow et al.
2. adversarial examples in the physical world, Kurakin et al.

[explaining and harnessing adversarial examples, Goodfellow et al.]: https://arxiv.org/abs/1412.6572
[adversarial examples in the physical world, Kurakin et al.]: http://arxiv.org/abs/1607.02533
