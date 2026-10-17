"""Zonotope helpers (geometry primitives -> zonotopes); robosuite/STL helpers of the research code removed."""
import os
from typing import List, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import einops

use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy:
    from zonopy.contset import zonotope, batchZonotope
else:
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope


def transform_zonotope(zono, pos, rot):
    """
    Transform 3D zonotope
    """
    assert isinstance(zono, zonotope) and zono.dimension == 3
    assert pos.shape[0] == 3

    rot = torch.asarray(rot, dtype = zono.dtype, device =zono.device)
    pos = torch.asarray(pos, dtype = zono.dtype, device =zono.device)

    c = zono.center
    G = zono.generators

    c_new = rot@c + pos
    G_new = G@rot.T
    Z_new = np.vstack([c_new, G_new])
    Z_new = torch.asarray(Z_new)

    zono_new = zonotope(Z_new)
    
    if hasattr(zono, "name"):
        zono_new.name = zono.name

    return zono_new

def scale_zonotope(zono, scale):
    """
    Scale 3D zonotope
    """
    assert isinstance(zono, zonotope) and zono.dimension == 3
    assert scale.shape[0] == 3
    
    c = zono.center
    G = zono.generators
    S = torch.diag(torch.tensor(scale)).to(dtype=zono.dtype, device=zono.device)
    
    G_new = S@G
    Z_new = np.vstack([c, G_new])
    
    Z_new = torch.asarray(Z_new)

    return zonotope(Z_new)

    

def plot_zonos(zonos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for zono in zonos:
        zono.plot3d(ax, facecolor='red', edgecolor='red')

    ax.set_xlim(-0.5, 0.5)  # set x-axis limits from 0 to 6
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.1)
    ax.view_init(elev=0, azim=0)
    
    fig.savefig('saved_figure1.png')
    

def get_zonotope_from_plane_geom(pos, rot, size):
    """
    Create zonotope from plane geometry

    The size means (half_width, half_height, spacing)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    size[2] = 0.05

    return get_zonotope_from_box_geom(pos, rot, size)

def get_zonotope_from_box_geom(pos, rot, size):
    """
    Create zonotope from box geometry

    The size means (half_dx, half_dy, half_dz)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag(size)

    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)


def get_zonotope_from_cylinder_geom(pos, rot, size):
    """
    Create zonotope from cylinder geometry

    The size means (radius, half_height)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag([size[0], size[0], size[1]])
    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)

def get_zonotope_from_sphere_geom(pos, rot, size):
    """
    Create zonotope from sphere geometry

    The size means (radius)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag([size[0], size[0], size[0]])
    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)

def get_zonotope_from_mesh_vertices(vertices):
    """
    Create zonotope that represents the bounding box of mesh ver
    """
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    c = (max_coords + min_coords)/2
    G = np.diag((max_coords - min_coords)/2)
    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)

def get_zonotope_from_segment(x1, x2):
    """
    Create zonotope that represents the line segment between x1 and x2
    """
    c = (x1 + x2)/2
    G = torch.abs((x2 - x1)/2)
    Z = torch.vstack([c, G])

    return zonotope(Z)

def maybe_flatten_the_zonotope(zonotope_list):
    assert isinstance(zonotope_list, list)
    
    flat_list = []
    
    def flatten(sublist):
        for item in sublist:
            if isinstance(item, batchZonotope):
                flatten(item)
            else:
                flat_list.append(item)
                
    flatten(zonotope_list)
    
    return flat_list

def get_bounding_box_of_zonotope_lists(zonotope_list: List[zonotope]):
    """
    Get the bounding box zonotope that contains all the zonotopes in the list

    Args:
        zonotope_list (list): list of zonotopes
    
    Returns:
        zonotope: the bounding box zonotope
    """
    V = torch.vstack([Z.polyhedron() for Z in zonotope_list])
    V = V.cpu().numpy()
    
    return get_zonotope_from_mesh_vertices(V)

def compute_collision_between_batchZono_and_Zono(batchZono: batchZonotope, zono: zonotope, combs = None):
    """
    Compute the collision constraints between batchZonotope and zonotope

    Args:
        links     : list of link represented by batchZonotope
        obstacles : list of obstacles represented by zonotope
        combs     : combinations used for polytope generation

    Returns:
        Cons      : constraint, negative means not colliding
        Jac       : TODO
    """
    if combs is None:
        combs = [torch.combinations(torch.arange(i), 2) for i in range(201)]

    zono_Z   = einops.repeat(zono.Z, 'm n -> repeat m n', repeat=batchZono.batch_shape[0])
    A_o, b_o = batchZonotope(torch.cat((zono_Z, batchZono.generators),-2)).polytope(combs) # A: n_timesteps,*,dimension  
    h_obs = (A_o@batchZono.center.unsqueeze(-1)).squeeze(-1) - b_o
    h_obs = h_obs.nan_to_num(-torch.inf)
    cons, ind = torch.max(h_obs, -1)
    
    cons = -cons

    return cons


if __name__ == "__main__":
    c = np.array([1, 1, 1])
    G = np.eye(3)
    Z = zonotope(np.vstack([c, G]))
    Z = torch.asarray(Z)

    Z_new = transform_zonotope(Z, pos=np.array([1, 2, 3]), rot=G)