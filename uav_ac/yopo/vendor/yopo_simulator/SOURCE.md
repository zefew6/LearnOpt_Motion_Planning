# Official YOPO simulator sources

Source: [TJU-Aerial-Robotics/YOPO](https://github.com/TJU-Aerial-Robotics/YOPO),
`YOPO-MINCO` revision `d5e1e60f83cc991602f142d9b6b96204a24d6b9f`,
upstream directory `Simulator/src`. The upstream MIT license is retained here.

`src/maps.cpp` and `src/perlinnoise.cpp` generate environments. The CUDA renderer,
its stereo dependencies, and the official collector are retained. ROS nodes,
RViz resources, demonstration images and duplicate configuration are excluded.
`assets/tree.ply` is an upstream model resource, not a generated dataset.

`provenance.json` records upstream and local hashes and excluded source resources.
Repository integration supplies explicit configuration, non-overwriting output
and early CUDA failure. Build artifacts and collection outputs belong in `runs/`.

```bash
cmake -S uav_ac/yopo/vendor/yopo_simulator -B runs/yopo/build -DYOPO_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runs/yopo/build --target dataset_generator -j2
```

For map export alone, configure with `-DYOPO_BUILD_CUDA=OFF` and build the
`map_generator` target. This local CLI calls the unchanged official map algorithms.
A PCL compatibility fix sets the generated room wall cloud dimensions before
`transformPointCloud`, preventing a division by zero in PCL 1.12.

Pure-UAV collection retains its official CUDA rendering contract. Whole-body
collection uses official map generation with the repository's articulated robot,
camera and whole-body collision interfaces.
