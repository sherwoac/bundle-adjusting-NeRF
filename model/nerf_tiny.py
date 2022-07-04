import numpy as np
import torch
import torch.nn.functional as torch_F
import camera
import tinycudann as tcnn

number_of_dimensions = 3
class NeRFTiny(torch.nn.Module):
    network_config = \
        {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": 256,
            "n_hidden_layers": 4,
        }

    rgb_encoding_config = \
        {
            "otype": "Frequency",
            "n_dims_to_encode": number_of_dimensions,
            "n_frequencies": 10
        }

    view_encoding_config = \
        {
            "otype": "Frequency",
            "n_dims_to_encode": number_of_dimensions,
            "n_frequencies": 4
        }

    def __init__(self, opt):
        super().__init__()
        self.layer_width = self.network_config["n_neurons"]
        self.define_network(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed
        self.k = torch.arange(opt.arch.posenc.L_3D, dtype=torch.float32, device=opt.device)

    def forward_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,density_samples = self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = density_samples*dist_samples # [B,HW,N]
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        return rgb,depth,opacity,prob # [B,HW,K]

    def define_network(self, opt):
        input_3D_dim = number_of_dimensions + 6 * opt.arch.posenc.L_3D if opt.arch.posenc else number_of_dimensions
        input_view_dim = number_of_dimensions + 6 * opt.arch.posenc.L_view if opt.arch.posenc else number_of_dimensions

        # point 3d
        # predict density and feature from 3d point location
        self.point_3d_encoding = tcnn.Encoding(number_of_dimensions, NeRFTiny.rgb_encoding_config, dtype=torch.float)

        # 63 x 128 in 4 layers
        self.point_3d_density_and_feature_1 = tcnn.Network(input_3D_dim, self.layer_width, NeRFTiny.network_config)

        # skip, (128 + 63) x 128 in 4 layers
        self.point_3d_density_and_feature_2 = tcnn.Network(self.layer_width + input_3D_dim,
                                                           self.layer_width + 1,
                                                           NeRFTiny.network_config)

        # view dependency
        # view dependent encoding, encode ray direction
        self.view_dependent_encoding = tcnn.Encoding(number_of_dimensions,
                                                     NeRFTiny.view_encoding_config,
                                                     dtype=torch.float)

        # view dependent MLP
        self.rgb_feature_direction_rgb = tcnn.Network(self.layer_width + input_view_dim,
                                                      3,
                                                      NeRFTiny.network_config)

    def get_positional_encoding_weightings(self, opt, L):
        start, end = opt.barf_c2f
        alpha = (self.progress.data-start)/(end-start)*L
        return (1-(alpha-self.k).clamp_(min=0, max=1).mul_(np.pi).cos_())/2


    def forward(self, opt, points_3D, ray_unit=None, mode=None):
        batch_size = points_3D.size()[0]
        sample_size = points_3D.size()[1]
        flattened_3d_points = points_3D.reshape((-1, points_3D.shape[-1]))
        flattened_directions = ray_unit.reshape((-1, ray_unit.shape[-1]))
        encoded_3d_points = self.point_3d_encoding(flattened_3d_points)
        if opt.barf_c2f is not None:
            weights = self.get_positional_encoding_weightings(opt, opt.arch.posenc.L_3D).repeat_interleave(2 * number_of_dimensions)
            encoded_3d_points = encoded_3d_points * weights

        points_and_encoded = torch.cat([flattened_3d_points, encoded_3d_points], dim=-1)
        first_features = self.point_3d_density_and_feature_1(points_and_encoded)
        features_and_encoded = torch.cat([first_features, points_and_encoded], dim=-1)
        second_features = self.point_3d_density_and_feature_2(features_and_encoded)
        depth = second_features[..., 0]
        rgb_features = second_features[..., 1:]
        encoded_directions = self.view_dependent_encoding(flattened_directions)
        directions_and_encoded = torch.cat([flattened_directions, encoded_directions], dim=-1)
        rgb_features_and_encoded_directions = torch.cat([rgb_features, directions_and_encoded], dim=-1)
        rgb = self.rgb_feature_direction_rgb(rgb_features_and_encoded_directions)
        rgb = rgb.reshape(batch_size, sample_size, -1, 3)
        depth = depth.reshape(batch_size, sample_size, -1)
        return rgb, depth
