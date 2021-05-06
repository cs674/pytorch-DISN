import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.models as models
# from torchvision.models.resnet import ResNet
# import torchvision.transforms as transforms
from torch.autograd import Variable

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
        self.local_feature_dim = 960
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

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.global_feature_dim + self.point_feature_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

        self.global_decoder = nn.Sequential(
            nn.Linear(self.global_feature_dim + self.point_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.local_decoder = nn.Sequential(
            nn.Linear(self.local_feature_dim + self.point_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    

    # x represents our data
    def forward(self, image_batch, points_batch, trans_mat=None):
        network_type = "two_stream"
        img_h = image_batch.shape[2]
        img_w = image_batch.shape[3]
        # print(image_batch.shape)
        # print("img_h {} img_w {}".format(img_h, img_w))

        if trans_mat != None:
            points_trans = torch.bmm(torch.cat((points_batch, torch.ones([points_batch.shape[0], points_batch.shape[1], 1], device=points_batch.device)), dim=-1), trans_mat)
            points_xy = points_trans[:, :, :2]/points_trans[:, :, 2].unsqueeze(-1)
            points_xy = torch.clamp(points_xy, 0, img_h-1)
            points_xy = torch.round(points_xy).long()

            # print('points_xy:', points_xy.shape)

            points_xy = points_xy.reshape(-1, 2)
            img_sample_points = torch.cat((torch.arange(0, points_batch.shape[0], device=points_batch.device).repeat_interleave(points_batch.shape[1]).unsqueeze(-1), points_xy), dim=-1)
            # print(img_sample_points.shape)


        # Get image features using resnet, 
        # layer = self.resnet._modules.get('avgpool')
        # self.resnet18.eval()
        layer_out = self.resnet18(image_batch) # can call each layer's feature use layer_out['layer1']
        global_img_features = layer_out['avgpool']

        # global_features = get_resnet_global_feature_vector(image_batch, self.resnet, layer, 512)
        if trans_mat != None:
            upsample = nn.Upsample(size=img_h,mode='bilinear')
            res_layer1 = upsample(layer_out['layer1'])
            points_res1 = res_layer1[img_sample_points[:, 0], :, img_sample_points[:, 1], img_sample_points[:, 2]].reshape([points_batch.shape[0], points_batch.shape[1], -1])
            res_layer2 = upsample(layer_out['layer2'])
            points_res2 = res_layer2[img_sample_points[:, 0], :, img_sample_points[:, 1], img_sample_points[:, 2]].reshape([points_batch.shape[0], points_batch.shape[1], -1])
            res_layer3 = upsample(layer_out['layer3'])
            points_res3 = res_layer3[img_sample_points[:, 0], :, img_sample_points[:, 1], img_sample_points[:, 2]].reshape([points_batch.shape[0], points_batch.shape[1], -1])
            res_layer4 = upsample(layer_out['layer4'])
            points_res4 = res_layer4[img_sample_points[:, 0], :, img_sample_points[:, 1], img_sample_points[:, 2]].reshape([points_batch.shape[0], points_batch.shape[1], -1])
            
            local_img_features = torch.cat((points_res1, points_res2, points_res3, points_res4), -1)
            
            # print('res_layer1: {}\n res_layer2:{}\n res_layer3:{}'.format(res_layer1.shape, 
            #     res_layer2.shape, res_layer3.shape))
            # print('resampled points: \n layer1:{}\nlayer2:{}\nlayer3:{}'.format(points_res1.shape,
            #     points_res2.shape, points_res3.shape))
            # print("points_img_features: {}".format(local_img_features.shape))
        
        
        # Get point features
        point_features = self.extract_point_features(points_batch)

        # repeat image_descriptors to be same shape as point_features
        global_img_features = global_img_features.squeeze().repeat(
            1, points_batch.shape[1]).reshape(points_batch.shape[0], points_batch.shape[1], self.global_feature_dim)

        # combine point features and image features
        global_features = torch.cat((global_img_features, point_features), -1)

        if trans_mat != None:
            local_features = torch.cat((local_img_features, point_features), -1)

        if trans_mat != None:
            sdf = self.global_decoder(global_features) + self.local_decoder(local_features)
        else:
            sdf = self.global_decoder(global_features)
        # print("pred sdf shape: {}".format(sdf.shape))
        

        return sdf
