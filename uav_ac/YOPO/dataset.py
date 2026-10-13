"""Validated numeric shards and explicitly separated privileged records."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from uav_ac.planning.end2end.contracts import DepthSpec, PlannerObservation

FIELDS=('depth_m','depth_valid','configuration','velocity','start_pva','goal')
DYNAMIC_FIELDS=FIELDS+('timestamp','start_jerk','derivative_source')


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('manifest path escapes dataset directory')
    return p


@dataclass(frozen=True)
class PrivilegedSample:
    scene_id: str
    scene_path: Path
    seed: int
    sample_index: int


class PlanningDataset:
    def __init__(self,path,*,split=None):
        self.path=Path(path).resolve()
        try:
            self.manifest=json.loads((self.path/'manifest.json').read_text())
            if self.manifest['schema_version'] not in (1,2):raise ValueError('unsupported dataset schema')
            fields=DYNAMIC_FIELDS if self.manifest['schema_version']==2 else FIELDS
            if split not in (None,'train','validation','test'):raise ValueError('unknown dataset split')
            self.spec=DepthSpec(**self.manifest['depth_spec'])
            self._records=[];self._shards={};self.scene_ids=[];seen=set();xml_paths=set();xml_hashes=set();shard_paths=set()
            for scene in self.manifest['scenes']:
                sid=scene['scene_id']
                if sid in seen:raise ValueError('duplicate scene_id across splits')
                seen.add(sid)
                if scene['split'] not in ('train','validation','test'):raise ValueError('unknown scene split')
                xml=_inside(self.path,scene['xml']);shard=_inside(self.path,scene['shard'])
                identity=file_hash(xml)
                if xml in xml_paths or identity in xml_hashes or shard in shard_paths:
                    raise ValueError('scene alias duplicates geometry/shard identity across splits')
                xml_paths.add(xml);xml_hashes.add(identity);shard_paths.add(shard)
                if identity!=scene['xml_sha256'] or file_hash(shard)!=scene['shard_sha256']:
                    raise ValueError('dataset file checksum mismatch')
                if self.manifest.get('profile')=='coupled_v3':
                    for resource in ('source_numeric','source_diagnostics'):
                        source=_inside(self.path,scene[resource])
                        if file_hash(source)!=scene[resource+'_sha256']:
                            raise ValueError('source witness checksum mismatch')
                if split is not None and split!=scene['split']:continue
                with np.load(shard,allow_pickle=False) as data:
                    if set(data.files)!=set(fields):raise ValueError('shard fields do not match observation schema')
                    arrays={name:data[name].copy() for name in fields}
                count=scene['count']
                if any(len(value)!=count for value in arrays.values()):raise ValueError('shard length mismatch')
                for i in range(count):PlannerObservation(**{k:v[i] for k,v in arrays.items()},depth_spec=self.spec)
                self.scene_ids.append(sid);self._shards[sid]=arrays
                self._records.extend((scene,i,xml) for i in range(count))
            robot=_inside(self.path,self.manifest['robot_xml'])
            if file_hash(robot)!=self.manifest['robot_sha256']:raise ValueError('robot model checksum mismatch')
        except (KeyError,OSError,TypeError,EOFError,json.JSONDecodeError) as exc:
            raise ValueError(f'invalid or incomplete dataset: {exc}') from exc

    def __len__(self):return len(self._records)
    def __getitem__(self,index):
        scene,i,xml=self._records[index];arrays=self._shards[scene['scene_id']]
        obs=PlannerObservation(**{k:v[i] for k,v in arrays.items()},depth_spec=self.spec)
        return obs,PrivilegedSample(scene['scene_id'],xml,scene['seed'],i)
