from pydantic import BaseModel, Field, computed_field

from .base import Myotubes, Nucleis, NucleiClusters


class InformationMetrics(BaseModel):
    """The information metrics of a MyoSam inference."""

    myotubes: Myotubes = Field(description="The myotubes.")
    nucleis: Nucleis = Field(description="The nucleis.")
    nuclei_clusters: NucleiClusters = Field(description="nuclei clusters.")

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{k}: {v if not isinstance(v, float) else round(v, 2)}"
                for k, v in self.model_dump().items()
                if k not in ["myotubes", "nucleis", "nuclei_clusters"]
            ]
        )

    def adjust_measure_unit(self, mu: float) -> None:
        """Adjust the measure unit of the metrics."""
        self.myotubes.adjust_measure_unit(mu)
        self.nucleis.adjust_measure_unit(mu)

    @computed_field  # type: ignore[misc]
    @property
    def total_myotubes(self) -> int:
        """The total number of myotubes."""
        return len(self.myotubes)

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis(self) -> int:
        """The total number of nuclei."""
        return len(self.nucleis)

    @computed_field  # type: ignore[misc]
    @property
    def total_myoblasts(self) -> int:
        """The total number of myoblasts."""
        return self.nucleis.num_myoblasts

    @computed_field  # type: ignore[misc]
    @property
    def total_nuclei_inside_myotubes(self) -> int:
        """The total number of nuclei inside myotubes."""
        return self.nucleis.num_nuclei_inside_myotubes

    @computed_field  # type: ignore[misc]
    @property
    def total_fusion_index(self) -> float:
        """The total fusion index."""
        return self.nucleis.total_fusion_index

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei_clusters(self) -> int:
        """The number of nuclei clusters."""
        return len(self.nuclei_clusters)

    @computed_field  # type: ignore[misc]
    @property
    def total_myotubes_area(self) -> float:
        """The total area of the myotubes."""
        return self.myotubes.area

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis_area(self) -> float:
        """The total area of the nuclei."""
        return self.nucleis.area

    @computed_field  # type: ignore[misc]
    @property
    def total_myoblasts_area(self) -> float:
        """The total area of the myoblasts."""
        return self.nucleis.myoblasts_area

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis_inside_myotubes_area(self) -> float:
        """The total area of the nuclei inside myotubes."""
        return self.nucleis.nucleis_inside_myotubes_area
