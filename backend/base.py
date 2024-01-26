from typing import Any
from functools import cached_property
from typing import Optional
import math
import statistics
from collections import defaultdict
import itertools
import networkx as nx

from pydantic import BaseModel, Field, computed_field, model_validator
import cv2
import numpy as np

from .utils import (
    object_overlaps_box,
    object_overlaps_polygon,
    object_overlaps_by_perc,
    vec_to_sym_matrix,
)


class MyoObject(BaseModel):
    """Base Class for MyoObjects: Myotubes and Nucleis."""

    identifier: int = Field(description="Identifier of the myoobject.")
    roi_coords: list[list[int]] = Field(description="ROI boundaries")  # (x, y)
    measure_unit: float = Field(description="Measure unit of the myoobject.")

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{k}: {v if not isinstance(v, float) else round(v, 2)}"
                for k, v in self.model_dump().items()
                if not k.startswith("roi_coords") and k != "rgb_repr"
            ]
        )

    @cached_property
    def roi_coords_np(self) -> np.ndarray:
        """ROI coordinates as a numpy array. (N, 1, 2)"""
        return np.array(self.roi_coords, dtype=np.int32)[:, None, :]

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> float:
        """Area of the myoobject."""
        return cv2.contourArea(self.roi_coords_np) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def convex_area(self) -> float:
        """Convex area of the myoobject."""
        return cv2.contourArea(self.convex_hull) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def solidity(self) -> float:
        """Solidity of the myoobject."""
        return self.area / self.convex_area

    @computed_field  # type: ignore[misc]
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the myoobject."""
        return self.elipse[1][0] / self.elipse[1][1]

    @computed_field  # type: ignore[misc]
    @property
    def roundness(self) -> float:
        """Roundness of the myoobject."""
        return 4 * self.area / (math.pi * self.elipse[1][0] ** 2)

    @computed_field  # type: ignore[misc]
    @property
    def perimeter(self) -> float:
        """Perimeter of the myoobject."""
        return cv2.arcLength(self.roi_coords_np, True) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def max_feret_diameter(self) -> float:
        """Feret's diameter of the myoobject."""
        return max(self.feret_bound_box)

    @computed_field  # type: ignore[misc]
    @property
    def min_feret_diameter(self) -> float:
        """Feret's diameter of the myoobject."""
        return min(self.feret_bound_box)

    @computed_field  # type: ignore[misc]
    @property
    def circularity(self) -> float:
        """Circularity of the myoobject."""
        return 4 * math.pi * self.area / (self.perimeter**2)

    @property
    def convex_hull(self) -> list[list[list[int]]]:
        """Convex hull of the myoobject."""
        return cv2.convexHull(self.roi_coords_np, returnPoints=True)

    @cached_property
    def feret_bound_box(self) -> list[int]:
        """Feret's bounding box of the myoobject."""
        conts = self.roi_coords_np.squeeze().astype(np.float32)
        _, eig_vecs = cv2.PCACompute(conts, mean=None)
        coords = np.matmul(eig_vecs, conts.T).T
        return cv2.boundingRect(coords)[2:]

    @cached_property
    def elipse(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """
        Elipse of the myoobject.

        Returns:
            ((center_x, center_y), (major_axis, minor_axis), angle)
        """
        return cv2.fitEllipse(self.roi_coords_np)


class Myotube(MyoObject):
    pred_iou: Optional[float] = Field(description="Predicted IoU")
    stability: Optional[float] = Field(description="Stability")
    rgb_repr: list[list[int]] = Field(
        description="RGB representation", exclude=True, default_factory=list
    )
    nuclei_ids: list[Optional[int]] = Field(
        description="Nucleis inside instance", default_factory=list
    )
    rgb_min: Optional[tuple] = Field(
        description="Minimum intensity of the myotube per channel",
        default_factory=tuple,
    )
    rgb_max: Optional[tuple] = Field(
        description="Maximum intensity of the myotube per channel",
        default_factory=tuple,
    )
    rgb_mean: Optional[tuple] = Field(
        description="Mean intensity of the myotube per channel",
        default_factory=tuple,
    )
    rgb_median: Optional[tuple] = Field(
        description="Median intensity of the myotube per channel",
        default_factory=tuple,
    )
    rgb_mode: Optional[tuple] = Field(
        description="Mode intensity of the myotube per channel",
        default_factory=tuple,
    )
    rgb_std: Optional[tuple] = Field(
        description="Standard deviation of the myotube per channel",
        default_factory=tuple,
    )
    integrated_density_rgb: Optional[tuple] = Field(
        description="Integrated density of the myotube per channel",
        default_factory=tuple,
    )

    @model_validator(mode="before")
    @classmethod
    def set_rgb_metrics(cls, values: Any) -> Any:
        """
        Sets the rgb metrics only on the initialisation afterwards we drop
        large rgb_repr lists to save on disk space.
        """
        if not isinstance(values, dict):
            if issubclass(values, BaseModel):
                values = values.model_dump()
            else:
                return values
        rgb_r = values.get("rgb_repr")
        if rgb_r:
            values["rgb_min"] = tuple(min(c) for c in zip(*rgb_r))
            values["rgb_max"] = tuple(max(c) for c in zip(*rgb_r))
            values["rgb_mean"] = tuple(statistics.mean(c) for c in zip(*rgb_r))
            values["rgb_median"] = tuple(
                statistics.median(c) for c in zip(*rgb_r)
            )
            values["rgb_mode"] = tuple(statistics.mode(c) for c in zip(*rgb_r))
            values["rgb_std"] = tuple(statistics.stdev(c) for c in zip(*rgb_r))
            values["integrated_density_rgb"] = tuple(
                sum(c) for c in zip(*rgb_r)
            )
        return values

    @computed_field  # type: ignore[misc]
    @property
    def instance_fusion_index(self) -> int:
        """Number of nuclei inside the myotube."""
        return len(self.nuclei_ids)

    @computed_field  # type: ignore[misc]
    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid of the myoobject. (x, y)"""
        return self.elipse[0]


class Nuclei(MyoObject):
    """A detected nuclei."""

    myotube_ids: list[Optional[int]] = Field(
        description="Identifer of the myotubes the nuclei belongs to."
    )
    centroid: tuple[float, float] = Field(
        description="Centroid of the nuclei. (x, y)"
    )
    prob: Optional[float] = Field(description="Probability of the nuclei pred")


class MyoObjects(BaseModel):
    """Base class for myotubes and myoblasts and other detected objects."""

    myo_objects: list[MyoObject] = Field(
        description="List of myoobjects.", default_factory=list
    )
    mapping: dict[int, list[Optional[int]]] = Field(
        description="Mapping of the myoobjects to other myoobjects.",
        default_factory=defaultdict,
        exclude=True,
    )

    def adjust_measure_unit(self, measure_unit: float) -> None:
        """Adjust the measure unit of the nucleis."""
        for myo in self.myo_objects:
            myo.measure_unit = measure_unit

    def move_object_to_end(self, idx: int) -> None:
        """Moves an object to the end of the list."""
        self.myo_objects.append(self.myo_objects.pop(idx))

    def filter_by_ids(self, ids: list[int]) -> "MyoObjects":
        """Filters the myoobjects by their ids."""
        return self.__class__(
            myo_objects=[m for m in self.myo_objects if m.identifier in ids]
        )

    def get_instance_by_point(
        self, point: tuple[int, int]
    ) -> Optional[MyoObject]:
        """Gets the instance id by x, y coordinates."""
        for myo in self.myo_objects:
            if cv2.pointPolygonTest(myo.roi_coords_np, point, False) >= 0:
                return myo
        return None

    @property
    def reverse_mapping(self) -> dict[Optional[int], list[int]]:
        """Reverse mapping of the myoobjects to other myoobjects."""
        reverse_mapping = defaultdict(list)
        for myo_id, myo_ids in self.mapping.items():
            for myo_id_ in myo_ids:
                reverse_mapping[myo_id_].append(myo_id)
        return reverse_mapping

    def __str__(self) -> str:
        return "area_total: {:.2f}".format(self.area)

    @property
    def area(self) -> float:
        """Area of the myoobjects."""
        return sum([m.area for m in self.myo_objects])

    def add_mapping(self, mapping: dict[int, list[Optional[int]]]) -> None:
        """Adds a mapping to the myoobjects."""
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.myo_objects)

    def __getitem__(self, idx: int) -> MyoObject:
        if isinstance(idx, slice):
            return self.__class__(myo_objects=self.myo_objects[idx])
        return self.myo_objects[idx]

    def __iter__(self):
        return iter(self.myo_objects)

    def __contains__(self, item: MyoObject) -> bool:
        return item in self.myo_objects


class Myotubes(MyoObjects):
    """The myotubes of a MyoSam inference."""

    myo_objects: list[Myotube] = Field(
        description="List of myotubes.", default_factory=list
    )

    def get_myotube_by_id(self, id: int) -> Myotube:
        return [m for m in self.myo_objects if m.identifier == id][0]

    def add_mapping(self, mapping: dict[int, list[Optional[int]]]) -> None:
        """Adds a mapping to the myoobjects."""
        super().add_mapping(mapping)
        for myo in self.myo_objects:
            myo.nuclei_ids = mapping[myo.identifier]


class Nucleis(MyoObjects):
    """The nucleis of a MyoSam inference."""

    myo_objects: list[Nuclei] = Field(
        description="List of nucleis.", default_factory=list
    )

    @property
    def num_myoblasts(self) -> int:
        """Number of myoblasts."""
        return len([m for m in self.myo_objects if not m.myotube_ids])

    @property
    def num_nuclei_inside_myotubes(self) -> int:
        """Number of myotubes."""
        return len(self) - self.num_myoblasts

    @property
    def myoblasts_area(self) -> float:
        """Area of the myoblasts."""
        return sum([m.area for m in self.myo_objects if not m.myotube_ids])

    @property
    def nucleis_inside_myotubes_area(self) -> float:
        """Area of the myotubes."""
        return self.area - self.myoblasts_area

    @property
    def total_fusion_index(self):
        return self.num_nuclei_inside_myotubes / self.num_myoblasts

    def __str__(self) -> str:
        return "\n".join(
            [
                "num_myoblasts: {}".format(self.num_myoblasts),
                "myoblasts_area: {:.2f}".format(self.myoblasts_area),
                "num_nucleis: {}".format(self.num_nuclei_inside_myotubes),
                "nucleis_area: {:.2f}".format(
                    self.nucleis_inside_myotubes_area
                ),
                "fusion_index: {:.2f}".format(self.total_fusion_index),
            ]
        )

    @classmethod
    def parse_nucleis(
        cls,
        roi_coords: np.ndarray,
        centroids: np.ndarray,
        myotubes: Myotubes,
        measure_unit: float = 1,
        probs: Optional[np.ndarray] = None,
    ) -> "Nucleis":
        """
        Parses the nucleis from the roi_coords and myotubes.
        Args:
            roi_coords (np.array): N x n_conts x 2 array of points. (x, y)
            centroids (np.array): N x 2 array of centroids. (x, y)
            probs (np.array): (N, ) array of probabilities.
            myotubes (Myotubes): The myotubes.
        """
        # np.flip(myoblast_rois.transpose(0, 2, 1), axis=2).astype(np.uint16)
        mapp = defaultdict(list)
        mapp_reverse = defaultdict(list)
        for myotube in myotubes:
            box = cv2.boundingRect(myotube.roi_coords_np)
            idx = np.where(object_overlaps_box(centroids[:, None, :], box))[0]
            if idx.size:
                msk = np.apply_along_axis(
                    object_overlaps_polygon,
                    -1,
                    centroids[idx],
                    myotube.roi_coords_np,
                )
                idx = idx[msk]
            if idx.size:
                msk = [
                    object_overlaps_by_perc(p, box, myotube.roi_coords_np)
                    for p in roi_coords[idx][:, :, None, :]
                ]
                for i in idx[msk]:
                    mapp[i].append(myotube.identifier)
                    mapp_reverse[myotube.identifier].append(i)
        nucleis = [
            Nuclei(
                identifier=i,
                roi_coords=coords,
                measure_unit=measure_unit,
                myotube_ids=mapp[i],
                centroid=centroids[i],
                prob=probs[i] if probs is not None else None,
            )
            for i, coords in enumerate(roi_coords)
        ]
        myotubes.add_mapping(mapp_reverse)
        return cls(myo_objects=nucleis, mapping=mapp)


class NucleiCluster(MyoObjects):
    """A detected nuclei cluster."""

    cluster_id: str = Field(description="Cluster identifier")
    myotube_id: int = Field(description="Myotube identifier")
    nuclei_ids: list[int] = Field(description="Nuclei identifiers")
    myo_objects: list[Nuclei] = Field(
        description="List of nucleis.", exclude=True, default_factory=list
    )

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei(self) -> int:
        """Number of nuclei."""
        return len(self.nuclei_ids)


class NucleiClusters(BaseModel):
    """The nuclei clusters of a MyoSam inference."""

    clusters: list[NucleiCluster] = Field(description="List of clusters.")

    def __str__(self) -> str:
        return f"num_clusters: {len(self.clusters)}"

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, idx: int) -> NucleiCluster:
        return self.clusters[idx]

    def __iter__(self):
        return iter(self.clusters)

    def get_clusters_by_myotube_id(self, idx: int) -> "NucleiClusters":
        """Gets the clusters by myotube id."""
        return self.__class__(
            clusters=[m for m in self.clusters if m.myotube_id == idx]
        )

    @classmethod
    def compute_clusters(cls, nucleis: Nucleis) -> "NucleiClusters":
        """Computes the clusters of nucleis."""
        mapping = nucleis.reverse_mapping
        nuclei_clusters = []
        for myo_id, nucleis_ids in mapping.items():
            if len(nucleis_ids) < 2:
                continue
            pairs = itertools.combinations(nucleis_ids, 2)
            lower_tri = np.array(
                [
                    object_overlaps_polygon(
                        nucleis[x].roi_coords_np.reshape(-1).astype(np.int16),
                        nucleis[y].roi_coords_np,
                    )
                    for x, y in pairs
                    if x is not None and y is not None
                ]
            )
            if not lower_tri.any():
                continue
            sym_matrix = vec_to_sym_matrix(lower_tri, len(nucleis_ids))
            graph = nx.from_numpy_array(sym_matrix)
            clusters = [
                list(x)
                for x in list(nx.connected_components(graph))
                if len(x) > 1
            ]
            for i, cluster in enumerate(clusters):
                cluster_id = str(myo_id) + "_" + str(i)
                nuclei_ids = [nucleis_ids[i] for i in cluster]
                clust = NucleiCluster(
                    cluster_id=cluster_id,
                    myotube_id=myo_id,
                    nuclei_ids=nuclei_ids,
                )
                nuclei_clusters.append(clust)
        return cls(clusters=nuclei_clusters)
