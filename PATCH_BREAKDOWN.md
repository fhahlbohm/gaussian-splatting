# FasterGS Patch Breakdown

Our fork applies some slight changes to the python frontend of Gaussian Splatting. Here is a detailed breakdown.

To evaluate changes and enable comparisons, our changes can be toggled with two global boolean variables (`USE_FASTERGS_RASTERIZER` and `USE_FASTERGS_ADAM`), which we will define here.

The following changes can also be seen in the github diff: 
```https://github.com/graphdeco-inria/gaussian-splatting/compare/54c035f...fhahlbohm:gaussian-splatting:e38cb1f```

### ``gaussian_renderer/__init__.py``

The improved rasterizer requires slightly different inputs that are easily obtained from the available data.
For improved readability, we added a separate `faster_render` function that is called by the original `render` function if the constant `USE_FASTERGS_RASTERIZER` at the top of the file is set to `True`.

```diff
# Top of file, import backend and set control variable for easy integration
+USE_FASTERGS_RASTERIZER = True 
+from FasterGSCudaBackend.torch_bindings import diff_rasterize, RasterizerSettings

# ....

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
            scaling_modifier = 1.0, separate_sh = False, override_color = None, 
            use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

# Add FasterGS rasterization, following original line 23
+    if USE_FASTERGS_RASTERIZER:
+        return faster_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp)

# .... rest of original render() function
# End of render() function 

# Added new function for readability showcasing the inputs of the rasterizer backend
# Add at end of file
+def faster_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False):
+    """
+    Render the scene.
+    Background tensor (bg_color) must be on GPU!
+    """
+    assert override_color is None, "FasterGSCudaBackend does not support the override_color argument."
+    assert pipe.compute_cov3D_python == False, "FasterGSCudaBackend does not support the compute_cov3D_python argument."
+    assert pipe.convert_SHs_python == False, "FasterGSCudaBackend does not support the convert_SHs_python argument."
+    assert pipe.debug == False, "FasterGSCudaBackend does not support the debug argument."
+
+    # Set up rasterization configuration.
+    raster_settings = RasterizerSettings(
+        w2c=viewpoint_camera.world_view_transform.T.contiguous(),  # FasterGSCudaBackend uses row-major matrices
+        cam_position=viewpoint_camera.camera_center.contiguous(),
+        bg_color=bg_color.contiguous(),
+        active_sh_bases=(pc.active_sh_degree + 1) ** 2,
+        width=int(viewpoint_camera.image_width),
+        height=int(viewpoint_camera.image_height),
+        focal_x=1 / math.tan(viewpoint_camera.FoVx / 2) * (viewpoint_camera.image_width / 2),
+        focal_y=1 / math.tan(viewpoint_camera.FoVy / 2) * (viewpoint_camera.image_height / 2),
+        center_x=viewpoint_camera.image_width / 2,
+        center_y=viewpoint_camera.image_height / 2,
+        near_plane=0.2,
+        far_plane=10000.0,
+        proper_antialiasing=pipe.antialiasing,
+    )
+
+    # In 1st dim rasterizer adds 1 when Gaussians are visible, in 2nd dim rasterizer adds gradients of 2d means when backward is called.
+    densification_info = torch.zeros((2, pc._xyz.shape[0]), dtype=torch.float32, device='cuda')
+
+    # Gather Gaussian parameters for rasterization.
+    means = pc._xyz
+    scales = pc._scaling if scaling_modifier == 1.0 else pc._scaling + math.log(max(scaling_modifier, 1e-6))
+    rotations = pc._rotation
+    opacities = pc._opacity
+    sh_coefficients_0 = pc._features_dc
+    sh_coefficients_rest = pc._features_rest
+
+    # Rasterize visible Gaussians to image.
+    rendered_image = diff_rasterize(
+        means=means,
+        scales=scales,
+        rotations=rotations,
+        opacities=opacities,
+        sh_coefficients_0=sh_coefficients_0,
+        sh_coefficients_rest=sh_coefficients_rest,
+        densification_info=densification_info,
+        rasterizer_settings=raster_settings,
+    )
+
+    # Apply exposure to rendered image (training only)
+    if use_trained_exp:
+        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
+        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
+
+    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
+    # They will be excluded from value updates used in the splitting criteria.
+    rendered_image = rendered_image.clamp(0, 1)
+    out = {
+        "render": rendered_image,
+        "densification_info": densification_info,
+        # "viewspace_points_grad": densification_info[1],
+        # "visibility_filter" : densification_info[0].nonzero(),
+    }
+
+    return out

```



### `scene/gaussian_model.py`

We slightly modified the optimizer setup so that the fused Adam optimizer can be toggled through the `USE_FASTERGS_ADAM` constant at the the top of the file.
  We also added a new helper function `add_densification_stats_fastergs`.

```diff 
# Top of file, import backend and set control variable for easy integration
+USE_FASTERGS_ADAM = True
+from FasterGSCudaBackend.torch_bindings import FusedAdam

# Add optional use of the fused Adam routine
# Originally lines 192-196, in training_setup(self, training_args):
        if self.optimizer_type == "default":
-            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
+            self.optimizer = FusedAdam(l, lr=0.0, eps=1e-15) if USE_FASTERGS_ADAM else torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

# FasterGS densification tracking
# Add at end of file
+    def add_densification_stats_fastergs(self, densification_info):
+        update_filter = densification_info[0] > 0
+        xyz_gradient = densification_info[1]
+        self.xyz_gradient_accum[update_filter] += xyz_gradient[update_filter, None]
+        self.denom[update_filter] += 1
```


### `train.py`

The Faster-GS rasterizer uses a modified input/output interface. Therefore, we modified the control flow to maintain compatibility and gracefully handle unsupported features.


```diff
# Originally line 16, import optional switch boolean
-from gaussian_renderer import render, network_gui
+from gaussian_renderer import render, network_gui, USE_FASTERGS_RASTERIZER
# ....

# Add use of FasterGS backend, dependant on switch boolean
# Originally line 112, in training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)
-        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
+        if USE_FASTERGS_RASTERIZER:
+            image, densification_info = render_pkg["render"], render_pkg["densification_info"]
+            viewspace_point_tensor, visibility_filter, radii = None, None, None
+        else:
+            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
+            densification_info = None

# ....

# Disable use of depth regularizer (not supported in base FasterGS)
# Originally line 130
-        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
+        if not USE_FASTERGS_RASTERIZER and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:

# ....

# Change use of densification
# Originally lines 164-173
            if iteration < opt.densify_until_iter:
-                # Keep track of max radii in image-space for pruning
-                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
-                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
+                if USE_FASTERGS_RASTERIZER:
+                    gaussians.add_densification_stats_fastergs(densification_info)
+                else:
+                    # Keep track of max radii in image-space for pruning
+                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
+                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
+                    radii = densification_info[0] if USE_FASTERGS_RASTERIZER else radii  # valid as only used for sparse adam visibility mask
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):


```