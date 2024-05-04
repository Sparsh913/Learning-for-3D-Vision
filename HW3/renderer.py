import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas, # deltas is the length of each ray segment
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        # pass
        # trans = torch.exp(-rays_density * deltas)
        # print("shape of deltas", deltas.shape) # (N, n_pts, 1)
        # trans = torch.ones_like(deltas.shape[0],)
        trans = torch.full(
            (deltas.shape[0],), 
            1.0, 
            device=deltas.device
            )
        # print("trans", trans)
        weights = []
        # print("rays_density", rays_density)HYDRA_FULL_ERROR=1
        # print("deltas", deltas)
        # print("rays density", rays_density)
        # print("deltas shape", deltas.shape)
        # print("deltas.shape[1]", deltas.shape[1])
        for i in range(deltas.shape[1]):
            # print("i", i)
            # print("trans", trans)
            weights.append(trans)
            trans = torch.mul(
                trans, 
                torch.exp(-torch.mul(rays_density[:, i, 0], 
                deltas[:,i, 0] + eps))
            )

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        # weights = torch.cat(weights, dim=-1)
        # print("weights", weights)
        weights = torch.stack(weights, dim=1).unsqueeze(-1) # (N, n_pts, 1)
        # print("shape of weights", weights.shape) # (N, n_pts, 1)
        # print("weights", weights)
        return weights * (1 - torch.exp(-rays_density * deltas + eps))
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # pass
        # print(" I am in aggregate")
        # print("shape of rays_feature", rays_feature.shape) # (N, n_pts, 3)
        # feature = torch.sum(weights.unsqueeze(-1) * rays_feature, dim=1)
        feature = torch.sum(weights * rays_feature, dim=1)
        # print("feature", feature)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            # print("curr_ray_bundle ----------------------------------- ", cur_ray_bundle.sample_lengths)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            # print("density", density)
            feature = implicit_output['feature'] # (N, n_pts, 3)
            # print("feature", feature)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            # print("depth_values", depth_values)
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # pass
            feature = self._aggregate(
                weights,
                feature.view(-1, n_pts, 3)
            )

            # TODO (1.5): Render depth map
            # pass
            # depth = torch.sum(weights * depth_values.view(-1, n_pts, 1), dim=1)
            # normalize the depth values by its maximum value
            depth_values = depth_values / depth_values.max()
            
            depth = self._aggregate(
                weights,
                depth_values.view(-1, n_pts, 1)
            )

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        # pass
        
        points = origins
        mask = torch.zeros(origins.shape[0], dtype=torch.bool, device=origins.device)
        for i in range(self.max_iters):
            dist = implicit_fn(points)
            mask = torch.abs(dist) < 1e-3
            points = points + dist * directions
            if mask.all():
                break
        return points, mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta, s = None):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    # pass
    # psi is the cumulative distribution function of the laplacian distribution with mean 0 and scale beta
    # print("signed distance", signed_distance.shape) # (N, 1)
    
    # psi = torch.zeros_like(signed_distance)
    # psi[signed_distance <= 0] = 0.5 * torch.exp(signed_distance[signed_distance <= 0] / beta)
    # psi[signed_distance > 0] = 1 - 0.5 * torch.exp(-signed_distance[signed_distance > 0] / beta)
    # density = psi * alpha
    
    if s is not None:
        return s * torch.exp(-s * signed_distance) / ((1 + torch.exp(-s * signed_distance))**2) 
    
    s = -signed_distance
    return alpha * (0.5 * torch.exp(s/beta) * (s <= 0) + (1 - 0.5 * torch.exp(-s/beta)) * (s > 0))

    # φs(x) = s*e^(−sx)/(1 + e^(−sx))^2
    

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            # density = None # TODO (Q7): convert SDF to density
            density = sdf_to_density(distance, self.alpha, self.beta)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
