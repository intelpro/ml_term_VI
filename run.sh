mode=generate
attack_mode=FGSM
dataset=CIFAR10
batch_size=100
epsilon=0.5
python main.py --attack_mode $attack_mode --mode $mode --dataset $dataset --batch_size $batch_size --epsilon $epsilon
