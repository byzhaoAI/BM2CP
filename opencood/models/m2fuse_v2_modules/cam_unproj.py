import torch

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2


def Mem2Ref(xyz_mem, bounds, voxel_size, Z, Y, X, assert_cube=False):
    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = bounds
    # xyz is B x N x 3, in mem coordinates transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)

    # get_mem_T_ref
    vox_size_X, vox_size_Y, vox_size_Z = voxel_size

    # translation. (this makes the left edge of the leftmost voxel correspond to XMIN)
    center_T_ref = torch.eye(4, device=torch.device(xyz_mem.device)).view(1,4,4).repeat([B, 1, 1])
    center_T_ref[:,0,3] = -XMIN-vox_size_X/2.0
    center_T_ref[:,1,3] = -YMIN-vox_size_Y/2.0
    center_T_ref[:,2,3] = -ZMIN-vox_size_Z/2.0

    # scaling. (this makes the right edge of the rightmost voxel correspond to XMAX)
    mem_T_center = torch.eye(4, device=torch.device(xyz_mem.device)).view(1,4,4).repeat([B, 1, 1])
    mem_T_center[:,0,0] = 1./vox_size_X
    mem_T_center[:,1,1] = 1./vox_size_Y
    mem_T_center[:,2,2] = 1./vox_size_Z
    mem_T_ref = torch.matmul(mem_T_center, center_T_ref)

    # get_ref_T_mem. note safe_inverse is inapplicable here, since the transform is nonrigid
    #先放到cpu上计算，完成后再放到gpu上参与下一步深度学习
    # ref_T_mem = mem_T_ref.inverse()
    ref_T_mem = torch.inverse(mem_T_ref.to('cpu')).to(mem_T_ref.device)
    xyz_ref = apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref


def gridcloud3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X
    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    # we want to sample for each location in the grid
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz


def unproject_image_to_mem(rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False, xyz_camA=None):
    # rgb_camB is B x C x H x W
    # pixB_T_camA is B x 4 x 4

    # rgb lives in B pixel coords
    # we want everything in A memory coords

    # this puts each C-dim pixel in the rgb_camB
    # along a ray in the voxelgrid
    B, C, H, W = list(rgb_camB.shape)

    # if xyz_camA is None:
    #     xyz_memA = gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)
    #     xyz_camA = Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)
    print(camB_T_camA.shape, xyz_camA.shape)
    xyz_camB = apply_4x4(camB_T_camA, xyz_camA)
    z = xyz_camB[:,:,2]

    xyz_pixB = apply_4x4(pixB_T_camA, xyz_camA)
    normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
    EPS=1e-6
    # z = xyz_pixB[:,:,2]
    xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
    # this is B x N x 2
    # this is the (floating point) pixel coordinate of each voxel
    x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]
    # these are B x N

    x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
    y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
    z_valid = (z>0.0).bool()
    valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

    if (0):
        # handwritten version
        values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
        for b in list(range(B)):
            values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
    else:
        # native pytorch version
        def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
            # make things in [-1,1]
            grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
            grid_x = 2.0*(grid_x / float(X-1)) - 1.0

            if clamp_extreme:
                grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
                grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

            return grid_y, grid_x

        y_pixB, x_pixB = normalize_grid2d(y, x, H, W)
        # since we want a 3d output, we need 5d tensors
        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False)

    values = torch.reshape(values, (B, C, Z, Y, X))
    values = values * valid_mem
    return values