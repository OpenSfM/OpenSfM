# pyre-strict
import os

import numpy as np
from opensfm import io
from opensfm.dataset import DataSet
from opensfm.dense import depthmap_to_ply, scale_down_image


def run_dataset(
    data: DataSet,
    no_cameras: bool,
    no_points: bool,
    depthmaps: bool,
    point_num_views: bool,
    cluster: int,
) -> None:
    """Export reconstruction to PLY format

    Args:
        no_cameras: do not save camera positions
        no_points: do not save points
        depthmaps: export per-image depthmaps as pointclouds
        point_num_views: Export the number of views associated with each point

    """

    reconstructions = data.load_reconstruction()
    tracks_manager = data.load_tracks_manager()
    no_cameras = no_cameras
    no_points = no_points
    point_num_views = point_num_views
    cluster_list=[cluster]
    if reconstructions:
        if cluster==-1:
           cluster_list=range(len(reconstructions))
        for _cluster_id in cluster_list:
            _name="reconstruction"
            if cluster==-1:
                _name+=str(_cluster_id)
            data.save_ply(
                reconstructions[int(_cluster_id)],
                tracks_manager,
                _name+".ply",
                no_cameras,
                no_points,
                point_num_views,
            )

    if depthmaps:
        udata = data.undistorted_dataset()
        urec = udata.load_undistorted_reconstruction()[0]
        for shot in urec.shots.values():
            rgb = udata.load_undistorted_image(shot.id)
            for t in ("clean", "raw"):
                path_depth = udata.depthmap_file(shot.id, t + ".npz")
                if not os.path.exists(path_depth):
                    continue
                depth = np.load(path_depth)["depth"]
                rgb = scale_down_image(rgb, depth.shape[1], depth.shape[0])
                ply = depthmap_to_ply(shot, depth, rgb)
                with io.open_wt(udata.depthmap_file(shot.id, t + ".ply")) as fout:
                    fout.write(ply)
