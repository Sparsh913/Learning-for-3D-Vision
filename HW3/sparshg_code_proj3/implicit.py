import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0] #[:,0, :] #[..., 0] # (N, 64)
        # print("depth_values in implicit ---------- ", depth_values) # [32768, 64]
        # print("ray_bundle.sample_lengths in implicit ---------- ", ray_bundle.sample_lengths[:2, :3, :4]) # [32768, 64, 64]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        # print("signed_distance", signed_distance)
        density = self._sdf_to_density(signed_distance)
        # print("density in implicit ------- ", density)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }
        # print("out['density']", out["density"])

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips, 
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
                
            # print("dimin", dimin)
            # print("dimout", dimout)

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            # print("y shape ---------------------------------------------------", y.shape)
            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
        dir_dep = False
    ):
        super().__init__()
        # print("cfg", cfg)
        # print("cfg.harmonic_functions_xyz", cfg.harmonic_functions_xyz)
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim
        # print("embedding_dim_xyz", embedding_dim_xyz)
        # print("embedding_dim_dir", embedding_dim_dir)

        # pass
        # self.
        # Nerf MLP for now can just take 3d position and output density and feature
        self.dir_dep = dir_dep
        self.mlp = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim= embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_xyz,
            skip_dim=0,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=[],
        )
        if self.dir_dep:
            # add the output of the xyz mlp with the dir mlp
            self.layer0 = nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
            self.mlp_dep = MLPWithInputSkips(
                n_layers= cfg.append_xyz[0], #cfg.n_layers_xyz,
                input_dim= cfg.n_hidden_neurons_xyz + embedding_dim_dir,
                output_dim=cfg.n_hidden_neurons_dir,
                skip_dim=0,
                hidden_dim=cfg.n_hidden_neurons_xyz,
                input_skips=[],
            )
            self.linear = nn.Linear(cfg.n_hidden_neurons_xyz, 3)
        else:
            self.linear = nn.Linear(cfg.n_hidden_neurons_xyz, 4) # 1 for density, 3 for feature
        
    def forward(self, ray_bundle):
        self.embedding_dim_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points.view(-1, 3))
        self.embedding_dim_dir = self.harmonic_embedding_dir(ray_bundle.directions.unsqueeze(0).repeat(ray_bundle.sample_points.size(1),1,1).view(-1, 3))
        print("self.embedding_dim_xyz", self.embedding_dim_xyz.shape)
        print("self.embedding_dim_dir", self.embedding_dim_dir.shape)
        y = self.mlp(self.embedding_dim_xyz, self.embedding_dim_xyz)
        print("y shape", y.shape)
        if self.dir_dep:
            y = self.layer0(y)
            concat = torch.cat((self.embedding_dim_dir, y), dim= -1)
            y1 = self.mlp_dep(concat, concat)
            y1 = self.linear(y1)
            dens = F.relu(y[:, 0].view(-1, 1))
            feature = F.sigmoid(y1)
        else:
            # y = self.mlp(self.embedding_dim_xyz, self.embedding_dim_xyz)
            y = self.linear(y)
            dens = F.relu(y[:, 0].view(-1, 1))
            feature = F.sigmoid(y[:, 1:])
        out = {
            'density': dens,
            'feature': feature,
        }
        
        return out

class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
        # color = True
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        
        # self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        # embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        # Define an MLP to output per-point SDF
        # self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(embedding_dim_xyz, 256))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(256, 512))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(512, 256))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(256, 128))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(128, 64))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(64, 32))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(32, 16))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(16, 8))
        # self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(8, 1))
        # self.layers.append(nn.LeakyReLU())
        
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        
        # if not color:
        self.mlp_dist = MLPWithInputSkips(
            n_layers=cfg.n_layers_distance,
            input_dim= embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_distance,
            skip_dim=0,
            hidden_dim=cfg.n_hidden_neurons_distance,
            input_skips=[],
        )
        self.linear_dist = nn.Linear(cfg.n_hidden_neurons_distance, 1)
            
        # else:
        self.mlp_color = MLPWithInputSkips(
            n_layers=cfg.n_layers_color,
            input_dim= embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_color,
            skip_dim=2,
            hidden_dim=cfg.n_hidden_neurons_color,
            input_skips=[2],
        )
        self.linear_color = nn.Linear(cfg.n_hidden_neurons_color, 3)
        
        
        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        # self.color_layers = nn.ModuleList()
        # self.color_layers.append(nn.Linear(embedding_dim_xyz, 256)) 
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(256, 512))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(512, 256))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(256, 128))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(128, 64))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(64, 32))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(32, 16))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(16, 8))
        # self.color_layers.append(nn.LeakyReLU())
        # self.color_layers.append(nn.Linear(8, 3))

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        # pass
        # use the mlp to get the distance
        
        y = self.mlp_dist(self.harmonic_embedding_xyz(points), self.harmonic_embedding_xyz(points))
        y = self.linear_dist(y)
        
        # y = torch.nn.LeakyReLU()(y)
        # y = self.harmonic_embedding_xyz(points)
        # for layer in self.layers:
        #     y = layer(y)
        return y
        
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        # points = points.view(-1, 3)
        # # pass
        # color = self.harmonic_embedding_xyz(points)
        # y = self.mlp_dist(self.harmonic_embedding_xyz(points), self.harmonic_embedding_xyz(points))
        y = self.harmonic_embedding_xyz(points)
        color = self.mlp_color(y, y)
        color = self.linear_color(color)
        color = torch.nn.Sigmoid()(color)
        # for layer in self.color_layers:
        #     color = layer(color)
        return color
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        distance = self.get_distance(points)
        color = self.get_color(points)
        return distance, color
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
