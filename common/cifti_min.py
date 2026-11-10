"""Minimal CIFTI-2 (NIfTI-2 + XML) reader: numpy + stdlib only.
read_cifti(path) -> (data[dim5, dim6], xml, dim); brain_models(xml) -> list of dicts (structure, model, offset, count, verts, ijk)."""
import numpy as np, struct, re
import xml.etree.ElementTree as ET
DT = {16: np.float32, 64: np.float64, 4: np.int16, 8: np.int32, 2: np.uint8}
def read_cifti(path, want_data=True):
    with open(path, 'rb') as f:
        hdr = f.read(540)
        if struct.unpack('<i', hdr[:4])[0] != 540: raise ValueError('not NIfTI-2 little endian')
        dtype = struct.unpack('<h', hdr[12:14])[0]
        dim = struct.unpack('<8q', hdr[16:80])
        vox_offset = struct.unpack('<q', hdr[168:176])[0]
        slope, inter = struct.unpack('<2d', hdr[176:192])
        ext = f.read(4)
        xml = None
        pos = 544
        while pos + 8 <= vox_offset:
            esize, ecode = struct.unpack('<2i', f.read(8))
            body = f.read(esize - 8); pos += esize
            if ecode == 32: xml = body.rstrip(b'\x00').decode('utf-8')
        data = None
        if want_data:
            f.seek(vox_offset)
            n5, n6 = dim[5], dim[6]
            data = np.frombuffer(f.read(n5 * n6 * np.dtype(DT[dtype]).itemsize), dtype=DT[dtype]).reshape(n6, n5).T  # (dim5, dim6)
            if slope not in (0.0, 1.0) or inter != 0.0: data = data * slope + inter
    return data, xml, dim
def brain_models(xml):
    root = ET.fromstring(xml)
    out = []
    for bm in root.iter('BrainModel'):
        a = bm.attrib
        vi = bm.find('VertexIndices')
        verts = np.array(vi.text.split(), int) if vi is not None else None
        vx = bm.find('VoxelIndicesIJK')
        ijk = np.array(vx.text.split(), int).reshape(-1, 3) if vx is not None else None
        out.append(dict(structure=a['BrainStructure'], model=a['ModelType'], offset=int(a['IndexOffset']), count=int(a['IndexCount']),
                        nvert=int(a.get('SurfaceNumberOfVertices', 0)), verts=verts, ijk=ijk))
    return out
