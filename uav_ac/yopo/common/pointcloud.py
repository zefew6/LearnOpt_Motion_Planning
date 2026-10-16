"""Scalar XYZ PLY reader shared by official UAV and whole-body map adapters."""
from pathlib import Path
import numpy as np


def read_points(path):
    """Read scalar vertex properties from official PCL binary or ASCII PLY."""
    types={'float':'f4','float32':'f4','double':'f8','float64':'f8',
           'uchar':'u1','uint8':'u1','char':'i1','int8':'i1',
           'int':'i4','int32':'i4','uint':'u4','uint32':'u4',
           'short':'i2','int16':'i2','ushort':'u2','uint16':'u2'}
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'ply':raise ValueError('expected official PLY map')
        properties=[];count=None;fmt=None;vertex=False
        while True:
            line=stream.readline()
            if not line:raise ValueError('incomplete PLY header')
            fields=line.decode('ascii').split()
            if not fields:continue
            if fields[0]=='end_header':break
            if fields[0]=='format':fmt=fields[1]
            if fields[0]=='element':
                vertex=fields[1]=='vertex'
                if vertex:count=int(fields[2])
            if fields[0]=='property' and vertex:
                if fields[1] not in types:raise ValueError('unsupported PLY vertex property')
                properties.append((fields[2],types[fields[1]]))
        if not count or not {'x','y','z'}<={p[0] for p in properties}:raise ValueError('map has no XYZ vertices')
        if fmt=='ascii':
            rows=np.array([[float(v) for v in stream.readline().split()] for _ in range(count)])
            points=rows[:,[[p[0] for p in properties].index(k) for k in ('x','y','z')]]
        elif fmt in ('binary_little_endian','binary_big_endian'):
            endian='<' if fmt=='binary_little_endian' else '>'
            data=np.fromfile(stream,dtype=np.dtype([(n,endian+t) for n,t in properties]),count=count)
            if len(data)!=count:raise ValueError('incomplete PLY vertices')
            points=np.column_stack([data[k] for k in ('x','y','z')])
        else:raise ValueError('unsupported PLY format')
    if not np.isfinite(points).all():raise ValueError('nonfinite map vertices')
    return points.astype(float)
