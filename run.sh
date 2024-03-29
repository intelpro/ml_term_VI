mode=ad_train # generate / train / test / ad_train / ad_test
load_ckpt_flag=True # if you want to train from scratch, flag should be false. True would be load weight file
ckpt_dir=checkpoints/CIFAR_vanila # loading ckpt directory name
env_name=CIFAR_vanila # ckpt save directory
dataset=CIFAR10 # MNIST or CIFAR10
network_choice=ResNet18 # ToyNet for MNIST ResNet18 for CIFAR10
attack_mode=ILLC #FGSM for FGSM, One-step target class method, Basic iterative method.  or ILLC for ILLC
target=-1 #-1 for non target, set any value btw 1-10 for target class method
iteration=1 #-1 for iterative mode(BIM, ILLC), 1 for one step mode(FGSM, one-step target class method)
batch_size=100 # default size is 100
learning_rate=5e-4 # default learning rate is 5e-4
epsilon=0.01 # epsilon pertubation, for adversarial train epsilon value will be redefined in function.
epoch=30
DEVICE=0
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --attack_mode $attack_mode --mode $mode --dataset $dataset --batch_size $batch_size --epsilon $epsilon --network_choice $network_choice --load_ckpt_flag $load_ckpt_flag --env_name $env_name  --lr $learning_rate --ckpt_dir $ckpt_dir --epoch ${epoch}
