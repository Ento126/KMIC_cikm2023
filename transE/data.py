from collections import Counter
from torch.utils import data
from typing import Dict, Tuple

Mapping = Dict[str, int]

def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class FB15KDataset(data.Dataset):
    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
