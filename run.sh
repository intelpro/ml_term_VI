mode=generate # generate or train or test or ad_train
load_ckpt_flag=True # if you want to train from scratch, flag should be false
ckpt_dir=checkpoints/CIFAR_vanila # loading ckpt directory name
attack_mode=ILLC #FGSM or ILLC
network_choice=ResNet18 # ToyNet ResNet18 or ResNet34 or ResNet50 or ResNet101
env_name=CIFAR_adv_train1 # ckpt save directory
dataset=CIFAR10 # MNIST or CIFAR10
batch_size=128
learning_rate=5e-4
epsilon=0.01
epoch=30
DEVICE=0
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --attack_mode $attack_mode --mode $mode --dataset $dataset --batch_size $batch_size --epsilon $epsilon --network_choice $network_choice --load_ckpt_flag $load_ckpt_flag --env_name $env_name  --lr $learning_rate --ckpt_dir $ckpt_dir --epoch ${epoch}
