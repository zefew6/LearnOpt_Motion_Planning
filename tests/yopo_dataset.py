"""Tiny, deterministic records for reader/training tests; no training sampler."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.simulation.depth_observation import IdealDepthObserver
from uav_ac.simulation.mujoco_sim import MujocoSimulation
from uav_ac.yopo.whole_body.contracts import DepthSpec, PlannerObservation
from uav_ac.yopo.whole_body.data.dataset import DYNAMIC_FIELDS, file_hash
from uav_ac.yopo.whole_body.validation.scenes import MODEL_ROOT, write_debug_scene


def write_dataset(output_dir, *, samples_per_scene=1, cylinder=False):
    """Write at most four hand-set observations in each of three test scenes."""
    assert 1 <= samples_per_scene <= 4
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / 'dataset'
    scenes = dataset / 'scenes'
    shards = dataset / 'shards'
    shards.mkdir(parents=True)
    (scenes / 'model').mkdir(parents=True)
    robot = scenes / 'model/aerial_manipulator.xml'
    shutil.copyfile(MODEL_ROOT / 'model/aerial_manipulator.xml', robot)
    entries = []
    spec = DepthSpec()
    for index, split in enumerate(('train', 'validation', 'test')):
        path = write_debug_scene(scenes, 'free')
        root = ET.parse(path).getroot()
        sid = f'scene_{index:03d}'
        root.set('model', sid)
        root.find('worldbody/body/include').set('file', 'model/aerial_manipulator.xml')
        if cylinder:
            ET.SubElement(root.find('worldbody'), 'geom', name='obstacle_bamboo_0',
                          type='cylinder', pos=f'2 {1.0 + index * .1} 1', size='.1 1')
        path = scenes / f'{sid}.xml'
        ET.ElementTree(root).write(path, encoding='unicode')
        sim = MujocoSimulation(path, planning_path_capacity=0, record_actual_trajectory=False)
        observations = []
        with IdealDepthObserver(sim.model, sim.data, spec=spec) as camera:
            for sample in range(samples_per_scene):
                derivatives = np.zeros((4, 8))
                derivatives[0, :3] = [.1 * sample, .05 * index, -1.8]
                derivatives[1, :3] = [.2, .01, 0.]
                derivatives[2, :3] = [.01, 0., 0.]
                q, v = recover_state(derivatives, .06)
                sim.robot.reset(q, v)
                depth, mask = camera.observe()
                goal = np.array([2., 0., -1.8, 0., 0., 0., 0., 0.])
                observations.append(PlannerObservation(
                    depth, mask, sim.robot.configuration, sim.robot.velocity,
                    derivatives[:3], goal, spec, 0., derivatives[3], 'augmentation'))
        shard = shards / f'{sid}.npz'
        np.savez_compressed(shard, **{name: np.stack([getattr(o, name) for o in observations])
                                     for name in DYNAMIC_FIELDS})
        entries.append(dict(scene_id=sid, family='free', split=split, seed=index,
                            xml=str(path.relative_to(dataset)), xml_sha256=file_hash(path),
                            shard=str(shard.relative_to(dataset)), shard_sha256=file_hash(shard),
                            count=samples_per_scene))
    manifest = dict(schema_version=2, profile='test_fixture', depth_spec=spec.as_mapping(),
                    gripper_gap=.06, robot_xml=str(robot.relative_to(dataset)),
                    robot_sha256=file_hash(robot), scenes=entries)
    path = dataset / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path
