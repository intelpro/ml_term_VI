mode=ad_train # generate or train or test or ad_train
load_ckpt_flag=True # if you want to train from scratch, flag should be false
attack_mode=ILLC #FGSM or ILLC
network_choice=ResNet18 # ResNet18 or ResNet34 or ResNet50 or ResNet101
env_name=ResNet18_ad_test # ckpt save directory
ckpt_dir=checkpoints/ResNet18_test # loading ckpt directory name
dataset=CIFAR10 # dataset to be used
batch_size=128
learning_rate=5e-4
epsilon=0.5
python main.py --attack_mode $attack_mode --mode $mode --dataset $dataset --batch_size $batch_size --epsilon $epsilon --network_choice $network_choice --load_ckpt_flag $load_ckpt_flag --env_name $env_name  --lr $learning_rate --ckpt_dir $ckpt_dir
