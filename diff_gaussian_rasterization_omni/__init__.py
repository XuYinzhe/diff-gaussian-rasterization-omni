from typing import NamedTuple, Optional, Tuple
from enum import IntEnum
import torch
import torch.nn as nn

from . import _C

class CameraModelType(IntEnum):
    PINHOLE = 1
    # FISHEYE = 2
    LONLAT = 3


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    RwcT: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    camera_type: int = CameraModelType.LONLAT
    render_depth: bool = False

def cpu_deep_copy_tuple(input_tuple):
    return tuple(
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    )

def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3Ds_precomp: torch.Tensor,
    raster_settings: GaussianRasterizationSettings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,       # (P, 3)   float32 CUDA
        means2D,       # (P, 3)   float32 CUDA – zeros placeholder for grad output
        sh,            # (P, K, 3) float32 CUDA  or empty
        colors_precomp,# (P, 3)   float32 CUDA  or empty
        opacities,     # (P, 1)   float32 CUDA
        scales,        # (P, 3)   float32 CUDA  or empty
        rotations,     # (P, 4)   float32 CUDA  or empty
        cov3Ds_precomp,# (P, 6)   float32 CUDA  or empty
        raster_settings: GaussianRasterizationSettings,
    ):
        # ---- Invoke the compiled CUDA forward rasterizer ----
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
            _C.rasterize_gaussians(
                raster_settings.bg,
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                int(raster_settings.camera_type),
                raster_settings.render_depth,
            )

        # ---- Save for backward ----
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )

        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii):
        # ---- Restore context ----
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings

        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # ---- Invoke the compiled CUDA backward rasterizer ----
        # The CUDA backward returns a tuple indexed as:
        #   [0] dL_dmeans2D
        #   [1] dL_dcolors
        #   [2] dL_dopacity
        #   [3] dL_dmeans3D
        #   [4] dL_dcov3D
        #   [5] dL_dsh
        #   [6] dL_dscales
        #   [7] dL_drotations
        backward_result = _C.rasterize_gaussians_backward(
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            int(raster_settings.camera_type),
        )

        dL_dmeans2D   = backward_result[0]
        dL_dcolors    = backward_result[1]
        dL_dopacity   = backward_result[2]
        dL_dmeans3D   = backward_result[3]
        dL_dcov3D     = backward_result[4]
        dL_dsh        = backward_result[5]
        dL_dscales    = backward_result[6]
        dL_drotations = backward_result[7]

        # Return gradients in the same order as forward() arguments:
        #   means3D, means2D, sh, colors_precomp, opacities,
        #   scales, rotations, cov3Ds_precomp, raster_settings
        return (
            dL_dmeans3D,    # grad for means3D
            dL_dmeans2D,    # grad for means2D
            dL_dsh,         # grad for sh
            dL_dcolors,     # grad for colors_precomp
            dL_dopacity,    # grad for opacities
            dL_dscales,     # grad for scales
            dL_drotations,  # grad for rotations
            dL_dcov3D,      # grad for cov3Ds_precomp
            None,           # no grad for raster_settings
        )

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (P, 3) float32 CUDA tensor of 3D means.

        Returns:
            Boolean tensor of shape (P,) indicating visibility.
        """
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            means3D:        (P, 3) 3D positions
            means2D:        (P, 3) placeholder for screen-space gradient output
            opacities:      (P, 1) opacity values
            shs:            (P, K, 3) spherical harmonics coefficients, or None
            colors_precomp: (P, 3) precomputed colors, or None
            scales:         (P, 3) scale factors, or None
            rotations:      (P, 4) rotation quaternions, or None
            cov3D_precomp:  (P, 6) precomputed 3D covariance, or None

        Returns:
            color: (3, H, W) rendered image
            radii: (P,) integer radii of each Gaussian in screen space
        """
        raster_settings = self.raster_settings

        # ---- Validate inputs ----
        if (shs is None and colors_precomp is None) or \
           (shs is not None and colors_precomp is not None):
            raise RuntimeError(
                "Please provide exactly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise RuntimeError(
                "Please provide exactly one of either scale/rotation pair "
                "or precomputed 3D covariance!"
            )

        # ---- Replace None with empty CUDA tensors ----
        if shs is None:
            shs = torch.empty(0, device="cuda")
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device="cuda")
        if scales is None:
            scales = torch.empty(0, device="cuda")
        if rotations is None:
            rotations = torch.empty(0, device="cuda")
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device="cuda")

        # ---- Rasterize ----
        color, radii = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

        return color, radii