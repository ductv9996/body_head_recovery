# data specifications
dir_image = "examples/face/image"
dir_mask = "../../dataset"
data_train = "places2"
data_test = "places2"
image_size = 512
mask_type = "pconv"

# model specifications
model = "aotgan"
block_num = 8
rates = [1, 2, 4, 8]
gan_type = "smgan"

# hardware specifications
seed = 2021
num_workers = 4

# optimization specifications
lrg = 0.0001
lrd = 0.0001
optimizer = "ADAM"
beta1 = 0.5
beta2 = 0.999

# loss specifications
rec_loss = "1*L1+250*Style+0.1*Perceptual"
adv_weight = 0.01

# training specifications
iterations = 1000000
batch_size = 8
port = 22334
resume = True

# log specifications
print_every = 10
save_every = 10000
save_dir = "../experiments"
tensorboard = True

# test and demo specifications
pre_train = "body_head_recovery/Inpainting/experiments/celebahq/G0000000.pt"
outputs = "outputs/"
thick = 15
painter = "freeform"