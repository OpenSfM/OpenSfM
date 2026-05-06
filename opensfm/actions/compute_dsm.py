# pyre-strict
import os

from opensfm import dataset, dsm
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, subfolder: str) -> None:
    """Compute a DSM from the fused point cloud.

    Args:
        subfolder: dataset's subfolder where undistorted data lives.
    """
    udata_path = os.path.join(data.data_path, subfolder)
    udataset = dataset.UndistortedDataSet(
        data, udata_path, io_handler=data.io_handler)
    reconstructions = udataset.load_undistorted_reconstruction()
    dsm.compute_dsm(udataset, reconstructions[0])
