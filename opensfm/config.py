# pyre-strict

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, List, Union

import yaml


@dataclass
class OpenSfMConfig:
    ##################################
    # Params for metadata
    ##################################
    use_exif_size: bool = True
    # Treat images from unknown camera models as coming from different cameras
    unknown_camera_models_are_different: bool = False
    # Default projection type to use when it cannot be inferred from EXIF metadata
    default_projection_type: str = "perspective"
    # Default focal length to sensor size ratio to use when it cannot be inferred from EXIF metadata
    default_focal_prior: float = 0.85
    # Enable downloading missing datum grids from the PROJ CDN
    proj_cdn_enabled: bool = True
    # Additional specific folder to load/store the geoid files and grids
    proj_grid_cache_dir: str = ""

    ##################################
    # Params for features
    ##################################
    # Feature type (AKAZE, SURF, SIFT, HAHOG, ORB, DSPSIFT)
    feature_type: str = "HAHOG"
    # If true, apply square root mapping to features
    feature_root: bool = True
    # If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.
    feature_min_frames: int = 4000
    # Same as above but for panorama images
    feature_min_frames_panorama: int = 16000
    # Resize the image if its size is larger than specified. Set to -1 for original size
    feature_process_size: int = 2048
    # Same as above but for panorama images
    feature_process_size_panorama: int = 4096
    feature_use_adaptive_suppression: bool = False
    # Bake segmentation info (class and instance) in the feature data. Thus it is done once for all at extraction time.
    features_bake_segmentation: bool = False
    # Maximum amount of memory to use for feature extraction (in MB). See default in features_processing.py.
    mem_ceiling: int | None = None
    # Ratio of the memory ceiling to use for feature extraction. See default in features_processing.py.
    mem_ratio: float | None = None

    ##################################
    # Params for SIFT
    ##################################
    # Smaller value -> more features
    sift_peak_threshold: float = 0.1
    # See OpenCV doc
    sift_edge_threshold: int = 10
    # See OpenCV doc
    sift_nfeatures: int = 0
    # See OpenCV doc
    sift_octave_layers: int = 3
    # See OpenCV doc
    sift_sigma: float = 1.6

    ##################################
    # Params for DSPSIFT
    ##################################
    dspsift_peak_threshold: float = 0.006
    dspsift_edge_threshold: int = 10

    ##################################
    # Params for SURF
    ##################################
    # Smaller value -> more features
    surf_hessian_threshold: float = 3000
    # See OpenCV doc
    surf_n_octaves: int = 4
    # See OpenCV doc
    surf_n_octavelayers: int = 2
    # See OpenCV doc
    surf_upright: int = 0

    ##################################
    # Params for AKAZE (See details in lib/src/third_party/akaze/AKAZEConfig.h)
    ##################################
    # Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    akaze_omax: int = 4
    # Detector response threshold to accept point
    akaze_dthreshold: float = 0.001
    # Feature type
    akaze_descriptor: str = "MSURF"
    # Size of the descriptor in bits. 0->Full size
    akaze_descriptor_size: int = 0
    # Number of feature channels (1,2,3)
    akaze_descriptor_channels: int = 3
    akaze_kcontrast_percentile: float = 0.7
    akaze_use_isotropic_diffusion: bool = False

    ##################################
    # Params for HAHOG
    ##################################
    hahog_peak_threshold: float = 0.00001
    hahog_edge_threshold: float = 10
    hahog_normalize_to_uchar: bool = True

    ##################################
    # Params for general matching
    ##################################
    # Ratio test for matches
    lowes_ratio: float = 0.85
    # FLANN, BRUTEFORCE, WORDS, OPENCL_HAMMING or OPENCL_BF
    matcher_type: str = "OPENCL_HAMMING"
    # Match symmetrically or one-way
    symmetric_matching: bool = True
    # Number of image pairs used to train the binary projection (OPENCL_HAMMING)
    binary_training_pairs: int = 100

    ##################################
    # Params for FLANN matching
    ##################################
    # Algorithm type (KMEANS, KDTREE)
    flann_algorithm: str = "KMEANS"
    # See OpenCV doc
    flann_branching: int = 8
    # See OpenCV doc
    flann_iterations: int = 10
    # See OpenCV doc
    flann_tree: int = 8
    # Smaller -> Faster (but might lose good matches)
    flann_checks: int = 20

    ##################################
    # Params for BoW matching
    ##################################
    bow_file: str = "bow_hahog_root_uchar_10000.npz"
    # Number of words to explore per feature.
    bow_words_to_match: int = 50
    # Number of matching features to check.
    bow_num_checks: int = 20
    # Matcher type to assign words to features
    bow_matcher_type: str = "FLANN"

    ##################################
    # Params for VLAD matching
    ##################################
    vlad_file: str = "bow_hahog_root_uchar_64.npz"

    ##################################
    # Params for guided matching
    ##################################
    # Number of randomized spanning-trees to samples over the tracks-graph
    guided_spanning_trees: int = 5
    # Random ratio higher bound edges are multiplied with
    guided_spanning_trees_random: float = 0.5
    # Threshold for epipolar distance for accepting a match in radians
    guided_matching_threshold: float = 0.006
    # Minimum track length for initial triangulation
    guided_min_length_initial: int = 3
    # Minimum track length for final triangulation
    guided_min_length_final: int = 3
    # Threshold of reprojection for extending a track within a new image (in radians)
    guided_extend_threshold: float = 0.002
    # Number of images considered as neighbors of another one
    guided_extend_image_neighbors: int = 50
    # Maximum number of reprojected neighbors (in the tracks-graph) to check when extending a track within a new image
    guided_extend_feature_neighbors: int = 10

    ##################################
    # Params for matching
    ##################################
    # Maximum gps distance between two images for matching
    matching_gps_distance: float = 150
    # Number of images to match selected by GPS distance. Set to 0 to use no limit (or disable if matching_gps_distance is also 0)
    matching_gps_neighbors: int = 0
    # Number of images to match selected by time taken. Set to 0 to disable
    matching_time_neighbors: int = 0
    # Number of images to match selected by image name. Set to 0 to disable
    matching_order_neighbors: int = 0
    # Number of images to match selected by BoW distance. Set to 0 to disable
    matching_bow_neighbors: int = 0
    # Maximum GPS distance for preempting images before using selection by BoW distance. Set to 0 to disable
    matching_bow_gps_distance: float = 0
    # Number of images (selected by GPS distance) to preempt before using selection by BoW distance. Set to 0 to use no limit (or disable if matching_bow_gps_distance is also 0)
    matching_bow_gps_neighbors: int = 0
    # If True, BoW image selection will use N neighbors from the same camera + N neighbors from any different camera. If False, the selection will take the nearest neighbors from all cameras.
    matching_bow_other_cameras: bool = False
    # Number of images to match selected by VLAD distance. Set to 0 to disable
    matching_vlad_neighbors: int = 0
    # Maximum GPS distance for preempting images before using selection by VLAD distance. Set to 0 to disable
    matching_vlad_gps_distance: float = 0
    # Number of images (selected by GPS distance) to preempt before using selection by VLAD distance. Set to 0 to use no limit (or disable if matching_vlad_gps_distance is also 0)
    matching_vlad_gps_neighbors: int = 0
    # If True, VLAD image selection will use N neighbors from the same camera + N neighbors from any different camera. If False, the selection will take the nearest neighbors from all cameras.
    matching_vlad_other_cameras: bool = False
    # Number of rounds to run when running triangulation-based pair selection
    matching_graph_rounds: int = 0
    # If True, removes static matches using ad-hoc heuristics
    matching_use_filters: bool = False
    # Use segmentation information (if available) to improve matching
    matching_use_segmentation: bool = False
    # Use orientation (if available) to improve matching
    matching_use_opk: bool = True
    # Maximum number of cross-component image pairs per pair of components matched exhaustively by the force_match_components pass. Above this, pairs are selected by VLAD similarity
    matching_components_exhaustive_cap: int = 100
    # Number of most similar images (by VLAD distance) each image of the smaller component is paired with in the other component, when above the exhaustive cap
    matching_components_vlad_neighbors: int = 5
    # Run an extra pass at the end of match_features that matches features across disconnected components of the matching graph
    force_match_components: bool = False

    ##################################
    # Params for geometric estimation
    ##################################
    # Outlier threshold for fundamental matrix estimation as portion of image width
    robust_matching_threshold: float = 0.004
    # Outlier threshold for essential matrix estimation during matching in radians
    robust_matching_calib_threshold: float = 0.004
    # Minimum number of matches to accept matches between two images
    robust_matching_min_match: int = 20
    # Outlier threshold for essential matrix estimation during incremental reconstruction in radians
    five_point_algo_threshold: float = 0.004
    # Minimum number of inliers for considering a two view reconstruction valid
    five_point_algo_min_inliers: int = 20
    # Number of LM iterations to run when refining relative pose during matching
    five_point_refine_match_iterations: int = 10
    # Number of LM iterations to run when refining relative pose during reconstruction
    five_point_refine_rec_iterations: int = 1000
    # Check for Necker reversal ambiguities. Useful for long focal length with long distance capture (aerial manned)
    five_point_reversal_check: bool = False
    # Ratio of triangulated points non-reversed/reversed when checking for Necker reversal ambiguities
    five_point_reversal_ratio: float = 0.95
    # Outlier threshold for accepting a triangulated point in radians
    triangulation_threshold: float = 0.006
    # Minimum angle between views to accept a triangulated point
    triangulation_min_ray_angle: float = 1.0
    # Minimum depth to accept a triangulated point
    triangulation_min_depth: float = 0.001
    # Triangulation type : either considering all rays (FULL), or sing a RANSAC variant (ROBUST)
    triangulation_type: str = "FULL"
    # Number of LM iterations to run when refining a point
    triangulation_refinement_iterations: int = 10
    # Outlier threshold for resection in radians
    resection_threshold: float = 0.004
    # Minimum number of resection inliers to accept it
    resection_min_inliers: int = 10

    ##################################
    # Params for track creation
    ##################################
    # Minimum number of features/images per track
    min_track_length: int = 2
    # Whether to use depth prior during BA
    use_depth_prior: bool = False
    # Depth prior default std deviation
    depth_std_deviation_m_default: float = 1.0
    # Whether depth is radial (distance to camera center) or Z value
    depth_is_radial: bool = False
    # Whether depth is stored as inverted depth
    depth_is_inverted: bool = False

    ##################################
    # Params for bundle adjustment
    ##################################
    # The standard deviation of the reprojection error
    reprojection_error_sd: float = 0.004
    # The standard deviation of the exif focal length in log-scale
    exif_focal_sd: float = 0.01
    # The standard deviation of aspect ratio, i.e. fu/fv, in log-scale
    aspect_ratio_sd: float = 0.01
    # The standard deviation of the principal point coordinates
    principal_point_sd: float = 0.01
    # The standard deviation of the first radial distortion parameter
    radial_distortion_k1_sd: float = 0.01
    # The standard deviation of the second radial distortion parameter
    radial_distortion_k2_sd: float = 0.01
    # The standard deviation of the third radial distortion parameter
    radial_distortion_k3_sd: float = 0.01
    # The standard deviation of the fourth radial distortion parameter
    radial_distortion_k4_sd: float = 0.01
    # The standard deviation of the first tangential distortion parameter
    tangential_distortion_p1_sd: float = 0.01
    # The standard deviation of the second tangential distortion parameter
    tangential_distortion_p2_sd: float = 0.01
    # The default horizontal standard deviation of the GCPs (in meters)
    gcp_horizontal_sd: float = 0.01
    # The default vertical standard deviation of the GCPs (in meters)
    gcp_vertical_sd: float = 0.1
    # Global weight for GCPs relative to regular observations (scaled by sqrt of avg tracks/shot)
    gcp_global_weight: float = 0.04
    # The standard deviation of GCP reprojection observations (in normalized image coordinates)
    gcp_observation_sd: float = 0.001
    # Annealing schedule for GCP weights: list of multipliers applied to gcp_global_weight
    # across successive bundle passes. Set to [1.0] to disable annealing (single pass).
    gcp_annealing_steps: List[float] = field(
        default_factory=lambda: [5.0, 25.0]
    )
    # The standard deviation of the rig translation
    rig_translation_sd: float = 0.1
    # The standard deviation of the rig rotation
    rig_rotation_sd: float = 0.1
    # Type of threshold for filtering outlier : either fixed value (FIXED) or based on actual distribution (AUTO)
    bundle_outlier_filtering_type: str = "FIXED"
    # For AUTO filtering type, projections with larger reprojection than ratio-times-mean, are removed
    bundle_outlier_auto_ratio: float = 3.0
    # For FIXED filtering type, projections with larger reprojection error after bundle adjustment are removed
    bundle_outlier_fixed_threshold: float = 0.006
    # Optimize internal camera parameters during bundle
    optimize_camera_parameters: bool = True
    # Optimize rig parameters during bundle
    optimize_rig_parameters: bool = False
    # Maximum optimizer iterations.
    bundle_max_iterations: int = 100
    # Weight threshold below which a point is considered an outlier and removed from the reconstruction
    bundle_outlier_weight_threshold: float = 0.5
    # Default ratio of outlier to inlier density peaks for IRLS mixture model (all error groups)
    bundle_irls_density_ratio: float = 0.001
    # Density ratio override for GCP 2D projection residuals (set higher to be more lenient on GCPs)
    bundle_irls_gcp_density_ratio: float = 0.00001
    # Density ratio override for GPS residuals (set higher to be more lenient on GPS)
    bundle_irls_gps_density_ratio: float = 0.00001

    # Ratio of (resection candidates / total tracks) of a given image so that it is culled at resection and resected later
    resect_redundancy_threshold: float = 0.7
    # Retriangulate all points from time to time
    retriangulation: bool = True
    # Retriangulate when the number of points grows by this ratio
    retriangulation_ratio: float = 1.2
    # Use analytic derivatives or auto-differentiated ones during bundle adjustment
    bundle_analytic_derivatives: bool = True
    # Bundle after adding 'bundle_interval' cameras
    bundle_interval: int = 999999
    # Bundle when the number of points grows by this ratio
    bundle_new_points_ratio: float = 1.2
    # Max image graph distance for images to be included in local bundle adjustment
    local_bundle_radius: int = 3
    # Minimum number of common points betwenn images to be considered neighbors
    local_bundle_min_common_points: int = 20
    # Max number of shots to optimize during local bundle adjustment
    local_bundle_max_shots: int = 30
    # Number of grid division for seleccting tracks in local bundle adjustment
    local_bundle_grid: int = 12
    # Number of grid division for selecting tracks in final bundle adjustment
    final_bundle_grid: int = 32
    # Min number of shots above which global bundle switches to the stochastic solver
    stochastic_bundle_shot_count: int = 4000
    # Max interior cameras optimized per stochastic round (N)
    stochastic_bundle_max_shots: int = 500
    # Number of seed shots fed to each stochastic round
    stochastic_bundle_random_shots: int = 20
    # How much of stochastic_bundle_max_shots, do we reserve for GCP-anchored shots
    stochastic_bundle_gcp_seeds_ratio: float = 0.5
    # Graph-hop radius for stochastic rounds (large so seeds can reach the N budget)
    stochastic_bundle_radius: int = 100
    # Upper bound on the number of stochastic rounds K (K is derived from scene size)
    stochastic_bundle_max_rounds: int = 100
    # For debugging purpose of large datasets: limit the maximum number of shots in incremental reconstruction
    incremental_max_shots_count: int = 0
    # Number of different pairs to try for incremental reconstruction bootstrap
    incremental_bootstrap_tries: int = 10
    # Average inlier ratio on resected shots to accept a bootstrap pair and grow it
    incremental_bootstrap_min_inliers_ratio: float = 0.80

    # Remove uncertain and isolated points from the final point cloud
    filter_final_point_cloud: bool = False

    # Save reconstructions at every iteration
    save_partial_reconstructions: bool = False

    ##################################
    # Params for GPS/GCP alignment
    ##################################
    # Use or ignore EXIF altitude tag
    use_altitude_tag: bool = True
    # orientation_prior or naive
    align_method: str = "auto"
    # horizontal, vertical or no_roll
    align_orientation_prior: str = "horizontal"
    # Enforce GPS position in bundle adjustment
    bundle_use_gps: bool = True
    # Enforce Ground Control Point position in bundle adjustment
    bundle_use_gcp: bool = True
    # Compensate GPS with a per-camera similarity transform
    bundle_compensate_gps_bias: bool = False
    # Thrershold for the reprojection error of GCPs to be considered outliers
    gcp_reprojection_error_threshold: float = 0.05

    ##################################
    # Params for rigs
    ##################################
    # Number of rig instances to use when calibration rigs
    rig_calibration_subset_size: int = 15
    # Ratio of reconstructed images needed to consider a reconstruction for rig calibration
    rig_calibration_completeness: float = 0.85
    # Number of SfM tentatives to run until we get a satisfying reconstruction
    rig_calibration_max_rounds: int = 10

    ##################################
    # Params for image undistortion
    ##################################
    # Format in which to save the undistorted images
    undistorted_image_format: str = "jpg"
    # Max width and height of the undistorted image
    undistorted_image_max_size: int = 100000

    ##################################
    # Params for depth estimation (PatchMatch OpenCL)
    ##################################
    # In OpenCL, ignore Intel GPU devices (they are too slow and buggy).  Set to False to use them anyway.
    opencl_ignore_intel_gpu_device: bool = True
    # Number of neighboring views considered as candidates
    depthmap_num_neighbors: int = 10
    # Number of neighboring views used for each depthmap
    depthmap_num_matching_views: int = 8
    # Minimum depth in meters.  Set to 0 to auto-infer from the reconstruction.
    depthmap_min_depth: float = 0
    # Maximum depth in meters.  Set to 0 to auto-infer from the reconstruction.
    depthmap_max_depth: float = 0
    # Maximum number of PatchMatch iterations
    depthmap_max_iterations: int = 4
    # Correlation patch window size (must be odd)
    depthmap_patch_size: int = 5
    # Maximum image dimension for processing (longer side)
    depthmap_max_image_size: int = 3200
    # Maximum PatchMatch cost to keep a pixel (0 = disabled)
    depthmap_max_cost: float = 0
    # Threshold to measure depth closeness (clean stage)
    depthmap_same_depth_threshold: float = 0.05
    # Min number of consistent views in clean stage
    depthmap_min_consistent_views: int = 3
    # Relative depth margin for space-carving votes; neighbor sees further by this fraction → carve vote.
    depthmap_carving_threshold: float = 0.01
    # Max carve votes a pixel can accumulate before being discarded.
    depthmap_max_carved_views: int = 1
    # Two-pass cleaning: pass 1 uses consistency only (no carving) on raw
    # depthmaps, pass 2 uses carving on the cleaned depthmaps from pass 1.
    depthmap_carving_two_pass: bool = True
    # Cosine threshold for grazing-angle detection (below → pixel is grazing, stricter filtering).
    depthmap_grazing_cos_threshold: float = 0.2
    # Save per-shot raw/clean PLYs and per-cluster debug PLYs (slow, for debugging only).
    depthmap_save_debug_ply: bool = False
    # ── Disk reclamation (intermediate depthmap/PLY cleanup) ──
    # Delete each raw depthmap (.raw.npz) once the whole cleaning phase has
    # produced its clean counterpart. Raw maps are only consumed by cleaning,
    # so this is lossless; disable to keep raw maps for inspection/re-cleaning.
    depthmap_delete_raw_after_clean: bool = True
    # Delete per-cluster fused_batch_*.ply once they are merged into fused.ply.
    depthmap_delete_fusion_batches: bool = True
    # Delete per-cluster DSM/ortho tiles once they are merged into the final DSM/ortho.
    depthmap_delete_geotiff_batches: bool = False
    # Additionally export the merged dense cloud as LAS / LAZ (next to fused.ply).
    dense_pointcloud_export_las: bool = True
    dense_pointcloud_export_laz: bool = True
    # Spatial sigma for bilateral NCC weighting
    depthmap_sigma_spatial: float = 1.5
    # Color sigma for bilateral NCC weighting, in normalized [0,1] intensity units.
    depthmap_sigma_color: float = 25.0 / 255.0
    # Use Census transform as fallback when bilateral NCC fails (low-texture regions).
    depthmap_use_census: bool = True
    # Number of multi-scale hierarchy levels (1 = full-res only, 2 = half+full,  3 = quarter+half+full, etc.).
    depthmap_hierarchy_levels: int = 3
    # Enable checkerboard bilateral median filter after PatchMatch iterations
    depthmap_checkerboard_filter: bool = False
    # Minimum connected component size in pixels; smaller segments are removed as speckle noise
    depthmap_speckle_min_size: int = 0
    # Maximum gap size in pixels for linear depth interpolation (0 = disabled)
    depthmap_gap_max_size: int = 0
    # Depth/normal smoothness weight for PatchMatch
    depthmap_smooth_weight: float = 0.2
    # Number of views from previous iteration that are forcibly kept in the view selection (anchoring).
    depthmap_anchor_views: int = 2
    # Weight for geometric consistency cost (0 = disabled). Applied per source view.
    depthmap_geom_consistency_weight: float = 0.05
    # Maximum number of reference views per cluster for geometric consistency.
    # Each ref gets at most depthmap_num_matching_views geom-consistency links,
    # so a cluster only needs to hold each ref's local neighbours; larger
    # clusters add no geom benefit and raise peak RAM. ~num_matching_views + margin.
    depthmap_cluster_max_size: int = 16
    # Spatial gate for clustering: drop covisibility edges whose camera baseline
    # exceeds this factor × the median baseline, so distant cameras that merely
    # see common ground don't scatter clusters across the scene. 0 disables.
    depthmap_cluster_edge_max_factor: float = 2.0
    # Hard cap on the total views (cluster refs + neighbours) loaded per cluster batch in the cleaning and fusion stages.  Bounds peak RAM.
    depthmap_max_cluster_views: int = 48
    # Use SfM points to seed a Delaunay planar prior before PatchMatch iterations
    depthmap_sfm_planar_prior: bool = False
    # Minimum baseline angle (degrees) for neighbor selection.
    depthmap_neighbor_min_angle: float = 3.0
    # Maximum baseline angle (degrees) for neighbor selection.
    depthmap_neighbor_max_angle: float = 60.0
    # SVO voxel resolution level, auto-derived from the cleaned depthmaps'
    # surface sampling. "fine" sets the voxel to the median per-pixel footprint
    # (depth/focal — the finest detail the data resolves); "half" doubles that
    # voxel (2x coarser); "quarter" quadruples it (4x coarser). One of
    # {"fine", "half", "quarter"}.
    depthmap_fusion_svo_voxel_level: str = "fine"
    # SVO truncation factor: truncation_distance = factor * voxel_size.
    depthmap_fusion_svo_trunc_factor: float = 8
    # SVO minimum weight for extracting points
    depthmap_fusion_svo_min_weight: float = 2.0
    # Number of multi-level fill passes (1=fine only, 3=fine+2 coarser).
    depthmap_fusion_svo_num_levels: int = 3
    # Flat-surface decimation factor for extraction (1=off, 4=keep 1/4). Reduces point density on flat surfaces while preserving sharp edges.
    depthmap_fusion_svo_decimate_flat: int = 2
    # Edge detection threshold for decimation (0-1). Crossings with normal divergence above this are considered edges and never decimated.
    depthmap_fusion_svo_edge_threshold: float = 0.15
    # Minimum observation count for both voxels in a zero-crossing.
    depthmap_fusion_svo_min_count: int = 2
    # Coverage-first view selection: minimum number of selected views that must observe each chunk cell before the per-chunk view budget (depthmap_max_cluster_views) is spent on quality. Guards against cells the SVO cannot accept (needs >= svo_min_count samples) or single-view outliers becoming completion holes. Set >= depthmap_fusion_svo_min_count; 1 = coverage only.
    depthmap_fusion_min_cell_observers: int = 2
    # Local adaptive weight threshold for extracting surfaces : allows weak areas to self-calibrate.
    depthmap_fusion_svo_relative_min_weight: float = 0.25
    # Maximum unique voxels per SVO sub-volume : clusters are spatially split so each piece stays within this budget.
    depthmap_fusion_svo_max_voxels: int = 80_000_000
    # Number of extra neighbor shots per cluster shot for fusion augmentation.
    # Adds views from outside the cluster to improve boundary quality.
    depthmap_fusion_svo_augment_neighbors: int = 4
    # Coarse grid cell size multiplier for pre-scan (cell = factor * voxel_size).
    depthmap_fusion_svo_coarse_factor: int = 16
    # Fusion chunk size: max coarse cells per chunk = the assignment granularity of the global KD-tree split (each chunk is fused by its best inverse-depth observers, points + DSM clipped to its disjoint core). 0 = auto (one GPU sub-volume's worth = svo_max_voxels / coarse_factor^3). Smaller = more, tighter chunks (smaller DSM tiles, tighter per-chunk view sets, more seams); larger = fewer chunks (bigger tiles, but the per-chunk view cap may drop coverage).
    depthmap_fusion_chunk_max_cells: int = 0
    # Photometric TSDF refinement
    depthmap_fusion_svo_refine_enabled: bool = False
    # Number of SDF refinement iterations.
    depthmap_fusion_svo_refine_iters: int = 50
    # Laplacian regularization weight (0 = disabled).
    depthmap_fusion_svo_refine_lambda_reg: float = 0.3
    # Surface-stabilization anchor weight: pulls the refined TSDF back toward the fused value (energy mu*(phi-phi0)^2)
    depthmap_fusion_svo_refine_lambda_anchor: float = 0.05
    # Early-stop: halt refinement once the per-iteration RMS surface motion falls below this fraction of its peak (the descent has converged).
    depthmap_fusion_svo_refine_early_stop_rel: float = 0.2
    # This caps how many fusers are kept alive on the GPU at once (each holds a hash table)
    depthmap_fusion_svo_bake_reuse_max_fusers: int = 4
    # Hard ceiling, as a fraction of device VRAM, on the bytes of retained fusers (hash table + refine images) kept resident
    depthmap_fusion_svo_bake_reuse_vram_fraction: float = 0.5
    # Extract a 3-D triangle mesh (Surface Nets / dual contouring of the fused  TSDF) alongside the fused point cloud.
    depthmap_fusion_mesh_enabled: bool = False
    # Delete the per-cluster mesh_batch_*.ply after merging into mesh.ply.
    depthmap_fusion_mesh_delete_batches: bool = True

    ##################################
    # Params for octree point cloud tiling (viewer streaming)
    ##################################
    # Maximum number of points stored in a single octree tile
    octree_max_points_per_tile: int = 50000
    # Maximum octree depth (root = 0)
    octree_max_depth: int = 15
    # Number of LOD representative points kept in each inner (non-leaf) tile
    octree_lod_sample_count: int = 10000
    # Out-of-core octree build: octree depth at which points are bucketed to disk (bucket count <= 8^depth).
    octree_split_depth: int = 4
    # Out-of-core octree build: clouds with more points than this are bucketed to disk so peak
    octree_max_bucket_points: int = 8_000_000

    ##################################
    # Params for Dense Colour Equalization (per-image exposure + white balance + vignetting)
    ##################################
    # Apply the saved per-image equalization to the source colour images during the dense colour bake (ortho + point cloud + mesh) when an equalization.json is present.
    equalize_apply_in_bake: bool = True
    # Highlight protection: the correction rolls off toward identity as the corrected luminance rises above this knee, so a brightening gain cannot "burn" already-bright regions (concrete, sky) to flat white. Lower protects more highlights; 255 disables (pure multiply, may clip).
    equalize_highlight_knee: float = 235.0
    # Number of radial vignetting coefficients (basis rho^2, rho^4, ...).
    equalize_vignette_order: int = 2
    # A track must be seen by at least this many reconstructed images to be used (fewer couples no images).
    equalize_min_track_length: int = 3
    # Cap on the number of observations fed to the solver (whole tracks are subsampled, deterministically, beyond it) — bounds memory/time on huge scenes.
    equalize_max_observations: int = 4000000
    # Drop measurements within this many levels of 0/255 (clipped → unreliable in log).
    equalize_saturation_margin: float = 2.0
    # IRLS (Huber) reweighting iterations for robustness to specular/moving outliers.
    equalize_irls_iterations: int = 5
    # Huber transition, in robust-sigma units (residuals beyond this are down-weighted).
    equalize_huber_delta: float = 2.0
    # Ridge on the vignetting coefficients (prior toward a flat lens; stabilises images with few high-radius observations).
    equalize_vignette_regularization: float = 0.1
    # Tiny ridge on the per-image gains (conditioning only).
    equalize_gain_regularization: float = 0.001

    ##################################
    # Params for DSM (Digital Surface Model)and ortho generation
    ##################################
    # Controls DSM + ortho rendering.
    dsm_enabled: bool = True
    # Crop the dense outputs (point cloud + DSM/ortho) to the convex hull of the SfM points on the ground plane, trimming the sparse fringe beyond the surveyed area. Only takes effect when dsm_enabled (the hull is computed by dense_clustering and consumed by dense_merging).
    dense_crop_to_sfm_hull: bool = True
    # Outlier trim before hulling: fraction (percent) cut off each extremity along each principal (PCA) axis of the SfM points, so a few stray points can't inflate the crop hull. 0 = plain convex hull of all points.
    dense_crop_percentile: float = 1.0
    # Robust-to-grazing depth clamp: in the fusion pre-scan, drop depth samples farther than this multiple of the view's median depth.
    dsm_territory_depth_factor: float = 2.0
    # Debug: also write each chunk's own DSM+ortho as georeferenced GeoTIFFs (dsm_cluster_XXXX.tif / ortho_cluster_XXXX.tif) before they are merged
    dsm_save_cluster_tiles: bool = True
    # Merge per-cluster DSM/ortho tiles by distance-transform feather blending
    dsm_merge_feather: bool = True
    # DSM/ortho feather margin (coarse cells): KD-tree chunks are disjoint, so each renders its DSM/ortho over its core cells dilated by this many coarse cells (XY only). The overlap gives the merge feather a band to cross-fade and the per-tile hole-fill neighbour context. Points/mesh stay clipped to the core (unique). 0 = no margin (hard seams).
    dsm_feather_margin_cells: int = 2
    # DSM/ortho completion footprint (coarse cells): the KD-tree partitions only reconstructed cells, so the empty interior of the scene (plazas, canopy gaps) belongs to no chunk. The 2D occupied plan is morphologically closed by this many coarse cells (then hole-filled) to define the completable footprint; each empty interior column is handed to its nearest chunk to own and fill. The closing is a PROPER (padded) closing bounded to the occupied bounding box: it bridges concavity mouths up to ~2x this width and (with fill_holes) encloses pockets, but does NOT balloon the footprint out to the rectangular bbox edges. So it can be raised to bridge wider gaps without runaway outward extension; it still fills genuine open bays up to ~2x this width.
    dsm_footprint_close_cells: int = 128
    # DSM/ortho completion footprint trim: before closing, cut each axis's sparse EXTREMITIES — leading/trailing rows/columns whose occupied-cell count is below this fraction of the axis's median (non-zero) count. This restrains the completion bounding box so a few outlier occupied cells can't inflate it and let the closing invent a wide sparse fringe. Cells outside the trimmed core are still rendered, just not completed around. 0 = off; 0.05-0.2 cuts thin tapering edges.
    dsm_footprint_trim_fraction: float = 0.1
    # Soft RAM budget (MB) for the DSM/ortho merge: the final raster is written band-by-band, sized so each band's accumulators stay under this
    dsm_merge_max_ram_mb: int = 512
    # Wall cull for the DSM mesh: a surface-net triangle is rasterized only if |cos| of its normal from vertical >= this
    dsm_wall_cull_nz: float = 0.5
    # Orthophoto color baking (svo_bake_colors): number of sharpest inlier views blended for the final color
    ortho_bake_n_final_views: int = 3
    # Tukey-biweight reweighting iterations for the robust color consensus.
    ortho_bake_irls_iterations: int = 5
    # Per-view horizon occlusion for hole-filled (interpolated) ortho cells: the bake marches the full DSM from each cell toward the camera and drops views blocked by a taller surface (roof)
    ortho_bake_dsm_occlusion: bool = True
    # Detail injection: recover the texture the multi-view blend low-passes away, without reintroducing seams.
    ortho_detail_injection: bool = True
    # Detail injection: Gaussian sigma (ortho pixels) splitting low/high bands.
    ortho_detail_sigma: float = 2.0
    # Detail injection: amount of high-pass added back (1.0 = full detail; >1 over-sharpens, <1 is gentler).
    ortho_detail_strength: float = 1.0
    # Gated 3x3 median despeckle of the baked ortho: a pixel is replaced by the local median only when it differs from it by more than this
    ortho_median_threshold: float = 24.0
    # DSM Post-process hole filling : a hole's connected component is "tiny" when it has <= hole_fill_small_area_max cells
    hole_fill_diffuse_iters: int = 1024
    # DSM Post-process hole filling : when a hole is "tiny", it is filled by diffusion
    hole_fill_small_area_max: int = 2 ** 18
    # DSM hole routing by hole SHAPE (not area, not boundary height): a hole is filled FLAT (low_flat occluded-ground model, sharp roof edge) only when it is COMPACT — its aspect ratio (major/minor principal axis of its cells, orientation-invariant) is <= this; more-elongated holes are filled by smooth edge-aware diffusion (follows the terrain) instead. This is the geometric guard against the "ridge canyon": an elongated hole running along a ridge/valley spans a wide altitude range, so flat-filling it at the lowest border altitude carves a flat canyon where that altitude is unrelated to the far end. Shape distinguishes a genuine occlusion STEP (compact man-made → flat is right) from terrain VARIATION (elongated → diffuse), which a boundary-height test cannot (a slope spans as much height as a roof). Independent of hole area, so the diffusion can be made strong enough (diffuse_iters / small_area_max) to fill large soil holes while compact man-made holes still get the sharp flat fill. 0 = disable the gate (route by hole_fill_small_area_max area instead, legacy). Lower = stricter (more diffusion); ~3-5 is typical.
    hole_fill_low_flat_max_aspect: float = 8.0
    # DSM low_flat THINNESS gate (complements the aspect gate above): a hole is only flat-filled if it is also THICK — its largest inscribed-disk radius (max of the distance transform = local half-width, in cells) is >= this; thinner holes are diffused instead. This catches morphologically THIN / feathery / branching holes that the bbox aspect ratio misses: a wiggly or branching feather has a near-round bounding box (low aspect → looks "compact") yet is thin everywhere, and flat-filling it carves a thin canyon. 0 = disable. Raise to diffuse thicker structures; ~2-4 cells is typical.
    hole_fill_low_flat_min_thickness: float = 15.0
    # DSM hole filling : pixel-closing iterations for the per-tile "local" enclosure footprint (bridges ragged render-boundary gaps before binary_fill_holes). Kept small: the closing is extensive (border_value=1), so a large value would balloon the footprint past the rendered surface. binary_fill_holes already fills every fully-enclosed void at any size; this only bridges open mouths.
    hole_fill_footprint_close: int = 32
    # DSM large-hole fill geometry: instead of linearly interpolating across the hole (which slants ground up to a bordering roof), complete it as a gravity-aligned FLAT surface at the LOW altitude of its boundary
    hole_fill_low_percentile: float = 20.0
    # Fallback occluded-ground heuristic (superseded by ortho_bake_dsm_occlusion, so 0 = off by default).
    hole_fill_occlusion_drop: float = 0.0
    # Coherence-enhancing shock filter on the DSM that sharpens edges
    dsm_shock_iterations: int = 6
    # Structure-tensor half-window in cells (larger = straighter / more coherent).
    dsm_shock_window: int = 5
    # Time step; keep <= 0.5 for stability.
    dsm_shock_dt: float = 0.25
    # Along-edge (tangential) diffusion weight; straightens the voxel-jittered boundary while the shock sharpens across it.
    dsm_shock_coherence: float = 0.1
    # Edge-strength gate: the shock only fires where the local slope (rise/run = height gradient / gsd) exceeds this
    dsm_shock_edge_slope: float = 2.0

    ##################################
    # Params for multi-processing/threading
    ##################################
    # Number of threads to use
    processes: int = 1
    # When processes > 1, number of threads used for reading images
    io_processes: int = 4

    ##################################
    # Params for submodel split and merge
    ##################################
    # Average number of images per submodel
    submodel_size: int = 80
    # Radius of the overlapping region between submodels
    submodel_overlap: float = 30.0
    # Relative path to the submodels directory
    submodels_relpath: str = "submodels"
    # Template to generate the relative path to a submodel directory
    submodel_relpath_template: str = "submodels/submodel_%04d"
    # Template to generate the relative path to a submodel images directory
    submodel_images_relpath_template: str = "submodels/submodel_%04d/images"

    ##################################
    # Params for report localization
    ##################################
    # Unit system for the quality report: "metric" or "imperial"
    report_unit_system: str = "metric"
    # Language for the quality report: "en", "fr", "es", "de", "it"
    report_language: str = "en"


def default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return asdict(OpenSfMConfig())


def load_config(filepath: str) -> Dict[str, Any]:
    """DEPRECATED: = Load config from a config.yaml filepath"""
    if not os.path.isfile(filepath):
        return default_config()

    with open(filepath) as fin:
        return load_config_from_fileobject(fin)


def load_config_from_fileobject(
    f: Union[IO[bytes], IO[str], bytes, str],
) -> Dict[str, Any]:
    """Load config from a config.yaml fileobject"""
    config = default_config()

    new_config = yaml.safe_load(f)
    if new_config:
        for k, v in new_config.items():
            config[k] = v

    return config
