import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.models as models
# from torchvision.models.resnet import ResNet
# import torchvision.transforms as transforms
from torch.autograd import Variable

def get_resnet_global_feature_vector(img_in, model, layer, layer_dim):
    features = torch.zeros([img_in.shape[0], layer_dim, 1, 1]).to(img_in.device)
    
    # copy the feature from the model's last pooling layer
    def copy_feature(m, i, o):
        print("resnet feature size: {}".format(o.data.shape))
        print(o.data[1, :, :, :])
        features.copy_(o.data)

    h = layer.register_forward_hook(copy_feature)
    
    model(img_in)
    
    h.remove()
    # Return the feature vector
    return features

class MyResNet(nn.Module):
    def __init__(self, layers, *args):
        super().__init__(*args)
        self.layers = layers
        self.resnet_layers_out = dict()
        self.fhooks = []
        self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        
        for i, l in enumerate(list(self.resnet18._modules.keys())):
            # print("keys in resnet: i : {}, l: {}".format(i, l))
            if i in self.layers:
                self.fhooks.append(getattr(self.resnet18, l).register_forward_hook(self.forward_hook(l)))
        
    def forward_hook(self, layer_name):
        def hook(module, data_in, data_out):
            self.resnet_layers_out[layer_name] = data_out
        return hook
    
    def forward(self, data_in):
        x = self.resnet18(data_in)
        return self.resnet_layers_out

class sdfnet(nn.Module):
    def __init__(self):

        super(sdfnet, self).__init__()
        
        # resnet
        self.global_feature_dim = 512
        self.point_feature_dim = 512
        self.resnet18 = MyResNet(layers = [4, 5, 6, 7, 8])
        
        # self.extract_image_features = nn.Sequential(
        #     nn.Conv2d(3, 64, 5, padding=2),
        #     nn.MaxPool2d(2),
        # )
        
        # self.create_global_image_descriptors = nn.AvgPool2d(68)

        self.extract_point_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.point_feature_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.global_feature_dim + self.point_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    

    # x represents our data
    def forward(self, image_batch, points_batch, trans_mat=None):
        network_type = "one_stream"
        img_h = image_batch.shape[2]
        img_w = image_batch.shape[3]
        print(image_batch.shape)
        print("img_h {} img_w {}".format(img_h, img_w))
        # Get image features using resnet, 
        # layer = self.resnet._modules.get('avgpool')
        # self.resnet.eval()
        layer_out = self.resnet18(image_batch) # can call each layer's feature use layer_out['layer1']
        global_features = layer_out['avgpool']
        # global_features = get_resnet_global_feature_vector(image_batch, self.resnet, layer, 512)
        if network_type == "one_stream":
            print("points_batch shape {}".format(points_batch.shape))
            points_batch_reshape = points_batch.unsqueeze(3)
            points_batch_reshape = torch.cat([points_batch_reshape, torch.zeros(points_batch.shape[0], 
                points_batch.shape[1], points_batch.shape[2], 1)], 3)
            print("reshaped points_batch: {}".format(points_batch_reshape.shape))
            upsample = nn.Upsample(size=img_h,mode='bilinear')
            res_layer1 = upsample(layer_out['layer1'])
            points_res1 = nn.functional.grid_sample(res_layer3, points_batch_reshape)
            res_layer2 = upsample(layer_out['layer2'])
            points_res2 = nn.functional.grid_sample(res_layer3, points_batch_reshape)
            res_layer3 = upsample(layer_out['layer3'])
            points_res3 = nn.functional.grid_sample(res_layer3, points_batch_reshape)
            
            
            print('res_layer1: {}\n res_layer2:{}\n res_layer3:{}'.format(res_layer1.shape, 
                res_layer2.shape, res_layer3.shape))
            print('resampled points: \n layer1:{}\nlayer2:{}\nlayer3:{}'.format(points_res1.shape,
                points_res2.shape, points_res3.shape))
            # print(res_layer2.shape)
        # size [32, 512, 5, 5] from layer4
        # size [32, 512, 1, 1] from avgpool
        # print("img_feature from resnet: {}".format(global_features.shape))
        
        

        # image_features = self.extract_image_features(image_batch)
        # print(image_features.shape)
        # image_descriptors = self.create_global_image_descriptors(image_features).squeeze()
        # print(image_descriptors.shape)

        image_descriptors = global_features.squeeze()
        # print("image_descriptors: {}".format(image_descriptors.shape))
        
        # Get point features
        point_features = self.extract_point_features(points_batch)
        # print(point_features.shape)

        # repeat image_descriptors to be same shape as point_features
        image_descriptors = image_descriptors.repeat(
            1, points_batch.shape[1]).reshape(points_batch.shape[0], points_batch.shape[1], self.global_feature_dim)
        # print(image_descriptors.shape)
        # print(image_descriptors[0, 0, :])
        # print(image_descriptors[0, -1, :])
        # print(image_descriptors[1, 0, :])
        # print(image_descriptors[1, -1, :])

        # combine point features and image features
        features = torch.cat((image_descriptors, point_features), -1)
        # print(features.shape)

        sdf = self.decoder(features)
        print("pred sdf shape: {}".format(sdf.shape))
        

        return sdf
