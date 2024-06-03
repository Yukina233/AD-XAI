from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_1, CIFAR10_LeNet_1_Autoencoder
from .Simple_Dense import Simple_Dense, Simple_Dense_Autoencoder
from .mlp import MLP, MLP_Autoencoder
from .custom_cnn_LeNet import SMD_Custom_LeNet, SMD_Custom_LeNet_Autoencoder, GHL_Custom_LeNet, \
    GHL_Custom_LeNet_Autoencoder
from .vae import VariationalAutoencoder
from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel


#注意此处与源码有不同
#源码是不同数据集有不同的网络结构(which is weird)
#注意bias必须要设为0,否则DeepSAD可能出现mode collapse(原论文中也提及)
h_dims = [128, 64]
rep_dim = 32

# h_dims = [512, 256]
# rep_dim = 128
def build_network(net_name, input_size ,ae_net=None):
    """Builds the neural network."""
    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    elif net_name == 'fmnist_LeNet':
        net = FashionMNIST_LeNet()

    elif net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    elif net_name == 'cifar10_LeNet_1':
        net = CIFAR10_LeNet_1()

    elif net_name == 'Dense':
        net = MLP(x_dim=input_size, h_dims=h_dims, rep_dim=rep_dim, bias=False)
    elif net_name == 'SMD_cnn':
        net = SMD_Custom_LeNet()
    elif net_name == 'GHL_cnn':
        net = GHL_Custom_LeNet()
    else:
        assert NotImplementedError
        # net = Simple_Dense(x_dim=input_size, h_dims=h_dims, rep_dim=rep_dim, bias=False)
    return net

def build_autoencoder(net_name, input_size):
    """Builds the corresponding autoencoder network."""
    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    elif net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    elif net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    elif net_name == 'cifar10_LeNet_1':
        ae_net = CIFAR10_LeNet_1_Autoencoder()

    elif net_name == 'Dense':
        ae_net = MLP_Autoencoder(x_dim=input_size, h_dims=h_dims, rep_dim=rep_dim, bias=False)
    elif net_name == 'SMD_cnn':
        ae_net = SMD_Custom_LeNet_Autoencoder()
    elif net_name == 'GHL_cnn':
        ae_net = GHL_Custom_LeNet_Autoencoder()
    else:
        assert NotImplementedError
        # ae_net = Simple_Dense_Autoencoder(x_dim=input_size, h_dims=h_dims, rep_dim=rep_dim, bias=False)
    return ae_net
