# machine learning term project 

### Dependencies
```
python 3.6.4
pytorch 1.0.1 post2
```

### Usage
1. running examples
modify run.sh file then run
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

