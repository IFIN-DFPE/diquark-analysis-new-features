import uproot
import awkward as ak

from tqdm.contrib.concurrent import thread_map


class DataLoader:
    def __init__(self, path_dict: dict[str, str], index_start=0, index_stop=None):
        self.default_branches = [
            "Jet",
            "Jet/Jet.PT",
            "Jet/Jet.Eta",
            "Jet/Jet.Phi",
            "Jet/Jet.BTag",
            "Particle/Particle.PID",
            "Particle/Particle.Status",
            "Particle/Particle.Mass",
            "Event/Event.Weight",
        ]
        self.path_dict = path_dict
        self.index_start = index_start
        self.index_stop = index_stop

    def filter_fbits(self, branches: list[str]) -> list[str]:
        """Filter out branch names containing 'fBits'."""
        return [b for b in branches if "fBits" not in b]

    def read_jet_delphes(self, filename: str, branches: list[str] = None) -> ak.Array:
        """Read a delphes output TTree from a ROOT file into an awkward array."""
        if branches is None:
            branches = self.default_branches

        with uproot.open(filename) as f:
            tree = f["Delphes"]
            branches = self.filter_fbits(branches)
            return tree.arrays(
                branches,
                library="ak",
                entry_start=self.index_start,
                entry_stop=self.index_stop,
            )

    def _load_dataset(self, key: str):
        arr = self.read_jet_delphes(self.path_dict[key])

        return key, arr

    def load_data(self) -> dict[str, ak.Array]:
        """Load all datasets specified in the path dictionary."""

        keys = list(self.path_dict.keys())

        # Load the datasets in parallel
        datasets = thread_map(
            self._load_dataset, keys, max_workers=64, desc="Loading data"
        )

        self.datasets = dict(datasets)
        assert len(self.datasets.keys()) == len(datasets), "Duplicate dataset key"

        return self.datasets
