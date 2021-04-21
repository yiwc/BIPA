import torch as th
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision
import torch
class TorchResnet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(TorchResnet, self).__init__(observation_space, features_dim)

        model_conv = torchvision.models.resnet50(pretrained=True)
        # for param in model_conv.parameters():
        #     param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 1000)

        # model_conv = model_conv.to(device)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = 3#observation_space.shape[2]# 0
        self.cnn = nn.Sequential(
            model_conv,
            # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)



        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 3#observation_space.shape[2]# 0
        self.cnn = nn.Sequential(

            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        #h-w-c => c-h-w
        # print(observations.shape)
        # observations=observations.permute(0,3,1,2)
        return self.linear(self.cnn(observations))

class DualEyeCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEyeCNN, self).__init__(observation_space, features_dim)



        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 3#observation_space.shape[2]# 0
        self.cnn1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        simgle_eye_space = gym.spaces.Box(0, 255, shape=(3, 128, 128), dtype='float32')
        s=128
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn1(
                th.as_tensor(simgle_eye_space.sample()[None]).float()
            ).shape[1]

        self.linear1 = nn.Sequential(
                                     nn.Linear(n_flatten, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU()
                                     )
        self.linear2 = nn.Sequential(
                                     nn.Linear(n_flatten, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU()
                                     )

        self.linear3 = nn.Sequential(
                                     nn.Linear(128*2, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, features_dim),
                                     nn.ReLU()
                                     )

        self.preprocess=nn.Sequential(
            torchvision.transforms.Normalize((128,128,128),(128,128,128))
        )
    def forward(self, observations: th.Tensor) -> th.Tensor:

        #h-w-c => c-h-w
        # print(observations.shape)
        # observations=observations.permute(0,3,1,2)
        # print(observations.shape)
        # obs1=observations

        observations = self.preprocess(observations)

        obs1 = observations[:, :, :, :128]
        obs2 = observations[:, :, :, 128:]


        c1=self.linear1(self.cnn1(obs1))
        c2=self.linear2(self.cnn2(obs2))

        out=self.linear3(th.cat((c1,c2),1))

        return out

class DualEyeResNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEyeResNet, self).__init__(observation_space, features_dim)

        res1 = torchvision.models.resnet50(pretrained=True)
        res1.fc = nn.Linear(res1.fc.in_features, 500)

        res2 = torchvision.models.resnet50(pretrained=True)
        res2.fc = nn.Linear(res2.fc.in_features, 500)

        n_input_channels = 3#observation_space.shape[2]# 0
        self.cnn1 = nn.Sequential(
            res1,
            nn.ReLU(),
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            res2,
            nn.ReLU(),
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # simgle_eye_space = gym.spaces.Box(0, 255, shape=(3, 128, 128), dtype='float32')
        # s=128

        self.linear1 = nn.Sequential(
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU()
                                     )
        self.linear2 = nn.Sequential(
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU()
                                     )

        self.linear3 = nn.Sequential(
                                     nn.Linear(128*2, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, features_dim),
                                     nn.ReLU()
                                     )

        self.preprocess=nn.Sequential(
            torchvision.transforms.Normalize((128,128,128),(128,128,128))
        )
    def forward(self, observations: th.Tensor) -> th.Tensor:

        #h-w-c => c-h-w
        # print(observations.shape)
        # observations=observations.permute(0,3,1,2)
        # print(observations.shape)
        # obs1=observations

        observations = self.preprocess(observations)

        obs1 = observations[:, :, :, :128]
        obs2 = observations[:, :, :, 128:]


        c1=self.linear1(self.cnn1(obs1))
        c2=self.linear2(self.cnn2(obs2))

        out=self.linear3(th.cat((c1,c2),1))

        return out

class DualEyeMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEyeMLP, self).__init__(observation_space, features_dim)

        # n_input_channels = 3#observation_space.shape[2]# 0
        self.obs_shape_len=observation_space.shape[0]

        std_n=64
        act=nn.Tanh
        self.extract_mlp=nn.Sequential(
            nn.Linear(int(self.obs_shape_len/2), std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
        )

        self.linear1 = nn.Sequential(
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act()
                                     )
        self.linear2 = nn.Sequential(
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act()
                                     )

        self.linear3 = nn.Sequential(
                                     nn.Linear(std_n*2, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, features_dim),
                                     act()
                                     )

    def forward(self, observations: th.Tensor) -> th.Tensor:

        obs1 = observations[:,:int(self.obs_shape_len/2)] # left eye
        obs2 = observations[:,int(self.obs_shape_len/2):] # right eye

        c1=self.linear1(self.extract_mlp(obs1))
        c2=self.linear2(self.extract_mlp(obs2))

        out=self.linear3(th.cat((c1,c2),1))

        return out

class DualEyeMLPez(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEyeMLPez, self).__init__(observation_space, features_dim)

        # n_input_channels = 3#observation_space.shape[2]# 0
        self.obs_shape_len=observation_space.shape[0]

        std_n=18
        act=nn.Tanh
        self.extract_mlp=nn.Sequential(
            nn.Linear(int(self.obs_shape_len/2), std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
        )

        self.linear1 = nn.Sequential(
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act()
                                     )
        self.linear2 = nn.Sequential(
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act()
                                     )

        self.linear3 = nn.Sequential(
                                     nn.Linear(std_n*2, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, features_dim),
                                     act()
                                     )

    def forward(self, observations: th.Tensor) -> th.Tensor:

        obs1 = observations[:,:int(self.obs_shape_len/2)] # left eye
        obs2 = observations[:,int(self.obs_shape_len/2):] # right eye

        c1=self.linear1(self.extract_mlp(obs1))
        c2=self.linear2(self.extract_mlp(obs2))

        out=self.linear3(th.cat((c1,c2),1))

        return out

class DualEyeTransformer(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEyeTransformer, self).__init__(observation_space, features_dim)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        tile=self.tile=int(features_dim/2)
        emsize = 2*tile  # embedding dimension
        nhid = 32  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = int(tile/2)   # the number of heads in the multiheadattention models
        dropout = 0  # the dropout value
        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]

        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.linear3 = nn.Sequential(
                                     nn.Linear(16, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, features_dim),
                                     act()
                                     )

        self.linear_trans = nn.Sequential(
            nn.Linear(8, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, int(features_dim )),
            act()
        )
        self.linear_target = nn.Sequential(
             nn.Linear(8, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )
        self.linear_fuse = nn.Sequential(
             nn.Linear(features_dim, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:
        # out=observations.repeat(1,1,self.tile)
        # out=self.transformer_encoder(out)
        # return out[:, 0, :]

        # out=torch.flatten(observations[:,1:,:],start_dim=1)
        # out = self.linear3(out)
        # return out


        l1 = torch.flatten(observations[:,1:3,:],1)
        l2 = torch.flatten(observations[:,3:5,:],1)
        r1 = torch.flatten(observations[:,5:7,:],1)
        r2 = torch.flatten(observations[:,7:9,:],1)

        vec_trans=torch.cat([l1,r1],1)
        vec_target=torch.cat([l2,r2],1)
        o_trans=self.linear_trans(vec_trans)
        o_target=self.linear_target(vec_target)
        o = o_trans * o_target
        # o = torch.cat([o_trans,o_target],1)
        o = self.linear_fuse(o)
        return o

        # o1=torch.dot([observations[:,3,:],observations[:,4,:]])
        # o2=torch.dot([observations[:,7,:],observations[:,8,:]])
        # # o=torch.cat([observations[:,3:5,:],observations[:,7:,:]],1)
        # o=self.linear3(torch.flatten(o,1,2))

        #
        # return torch.ones([observations.shape[0],256]).to("cuda:0")

class DualEzMap(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualEzMap, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear3 = nn.Sequential(
                                     nn.Linear(16, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, std_n),
                                     act(),
                                     nn.Linear(std_n, features_dim),
                                     act()
                                     )

        self.linear_trans = nn.Sequential(
            nn.Linear(8, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, std_n),
            act(),
            nn.Linear(std_n, int(features_dim )),
            act()
        )

        self.linear_fuse = nn.Sequential(
             nn.Linear(features_dim, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.linear_at = nn.Sequential(
            nn.Linear(2,8),
            act(),
            nn.Linear(8,8),
            act(),
            nn.Linear(8,2),
            act(),
        )
        self.linear_map = nn.Sequential(
            nn.Linear(4,8),
            act(),
            nn.Linear(8,8),
            act(),
            nn.Linear(8,4),
            act(),
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = torch.flatten(observations[:,1:3,:],1)
        l2 = observations[:,3:5,:]
        r1 = torch.flatten(observations[:,5:7,:],1)
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]
        la = self.linear_at(lt.squeeze()).reshape([-1,1,2]) # ? 1 2
        ra = self.linear_at(rt.squeeze()).reshape([-1,1,2]) # ? 1 2

        lmap = self.linear_map(l1).reshape([-1,2,2]) # ? 2 2
        rmap = self.linear_map(r1).reshape([-1,2,2]) # ? 2 2

        la_=torch.matmul(la,lmap) # ? 2 2
        ra_=torch.matmul(ra,rmap) # ? 2 2
        a= la_+ra_
        o= a.squeeze().repeat(1,64)

        # vec_trans=torch.cat([l1,r1],1)
        # vec_target=torch.cat([l2,r2],1)
        # o_trans=self.linear_trans(vec_trans)
        # o_target=self.linear_target(vec_target)
        # o = o_trans * o_target
        # # o = torch.cat([o_trans,o_target],1)
        # o = self.linear_fuse(o)
        return o

        # o1=torch.dot([observations[:,3,:],observations[:,4,:]])
        # o2=torch.dot([observations[:,7,:],observations[:,8,:]])
        # # o=torch.cat([observations[:,3:5,:],observations[:,7:,:]],1)
        # o=self.linear3(torch.flatten(o,1,2))

        #
        # return torch.ones([observations.shape[0],256]).to("cuda:0")

class DualMap(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualMap, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )



    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        obs=torch.cat([l1,r1,lt,rt],1).flatten(1)

        o = self.linear(obs)
        # o= a.squeeze().repeat(1,64)

        # vec_trans=torch.cat([l1,r1],1)
        # vec_target=torch.cat([l2,r2],1)
        # o_trans=self.linear_trans(vec_trans)
        # o_target=self.linear_target(vec_target)
        # o = o_trans * o_target
        # # o = torch.cat([o_trans,o_target],1)
        # o = self.linear_fuse(o)
        return o

        # o1=torch.dot([observations[:,3,:],observations[:,4,:]])
        # o2=torch.dot([observations[:,7,:],observations[:,8,:]])
        # # o=torch.cat([observations[:,3:5,:],observations[:,7:,:]],1)
        # o=self.linear3(torch.flatten(o,1,2))

        #
        # return torch.ones([observations.shape[0],256]).to("cuda:0")

class DualMapV2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualMapV2, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.ab_extractor_linear = nn.Sequential(
             nn.Linear(6, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]


        #a <- lt,l1
        a= self.ab_extractor_linear(torch.cat([l1,lt],1).flatten(1))
        b= self.ab_extractor_linear(torch.cat([r1,rt],1).flatten(1))
        o= torch.cat([a,b],1)
        #b <- rt,r1
        # o = cat (a,b)

        # obs=torch.cat([l1,r1,lt,rt],1).flatten(1)
        # o = self.linear(obs)
        return o
class DualMapV3(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DualMapV3, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.a_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )
        self.b_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        #a <- lt multi l1^-1
        try:
            a=self.a_extractor_linear(torch.matmul(lt,torch.inverse(l1)).flatten(1))
        except:
            print(l1)
        b=self.b_extractor_linear(torch.matmul(rt,torch.inverse(r1)).flatten(1))

        o= torch.cat([a,b],1)
        return o


class IML(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(IML, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.a_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )
        self.b_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        #a <- lt multi l1^-1

        # try:
        with torch.no_grad():
            l1=l1+torch.rand_like(l1)*0.00001
            l2=l2+torch.rand_like(l2)*0.00001
            # rk1 = torch.matrix_rank(l1).item()
            # rk2 = torch.matrix_rank(l1).item()
            # if rk1<2 :
            #     l1=torch.eye(2)
            # if rk2<2 :
            #     l2=torch.eye(2)


        a=self.a_extractor_linear(torch.matmul(lt,torch.inverse(l1)).flatten(1))
        # except:
        # a=self.a_extractor_linear(torch.matmul(lt,l1).flatten(1))

            # print(l1)
        # try:
        b=self.b_extractor_linear(torch.matmul(rt,torch.inverse(r1)).flatten(1))
        # except:
        # b=self.b_extractor_linear(torch.matmul(rt,r1).flatten(1))
        o= torch.cat([a,b],1)
        return o

class NM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(NM, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(8, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.a_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )
        self.b_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        o = self.linear(
            torch.cat([l2[:,0:2,:].flatten(1),r2[:,0:2,:].flatten(1)],1).flatten(1)
        )

        return o

class PML(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(PML, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(8, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.ab_extractor_linear = nn.Sequential(
             nn.Linear(6, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )



    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        a= self.ab_extractor_linear(torch.cat([l1,lt],1).flatten(1))
        b= self.ab_extractor_linear(torch.cat([r1,rt],1).flatten(1))
        o= torch.cat([a,b],1)

        return o

class MML(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(MML, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.a_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )
        self.b_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        #a <- lt multi l1^-1
        try:
            a=self.a_extractor_linear(torch.matmul(lt,l1).flatten(1))
        except:
            print(l1)
        b=self.b_extractor_linear(torch.matmul(rt,r1).flatten(1))

        o= torch.cat([a,b],1)
        return o

class MonIML(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(MonIML, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.obs_shape_len=observation_space.shape[0]


        self.linear = nn.Sequential(
             nn.Linear(12, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )

        self.a_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim)),
             act()
        )
        self.b_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )


    def forward(self, observations: th.Tensor) -> th.Tensor:

        l1 = observations[:,1:3,:]
        l2 = observations[:,3:5,:]
        r1 = observations[:,5:7,:]
        r2 = observations[:,7:9,:]

        lt = l2[:,0:1,:] - l2[:,1:2,:]
        rt = r2[:,0:1,:] - r2[:,1:2,:]

        #a <- lt multi l1^-1
        try:
            a=self.a_extractor_linear(torch.matmul(lt,torch.inverse(l1)).flatten(1))
        except:
            a=self.a_extractor_linear(torch.matmul(lt,l1).flatten(1))
        # b=self.b_extractor_linear(torch.matmul(rt,torch.inverse(r1)).flatten(1))

        o= torch.cat([a],1)
        return o
