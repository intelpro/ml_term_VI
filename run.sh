mode=train # generate or train or test
load_ckpt_flag=False # if you want to train from scratch, flag should be false
attack_mode=FGSM #IterativeLeast or FGSM
network_choice=ResNet34 # ResNet18 or ResNet34 or ResNet50 or ResNet101
dataset=CIFAR10
batch_size=100
epsilon=0.5
python main.py --attack_mode $attack_mode --mode $mode --dataset $dataset --batch_size $batch_size --epsilon $epsilon --network_choice $network_choice --load_ckpt_flag $load_ckpt_flag
