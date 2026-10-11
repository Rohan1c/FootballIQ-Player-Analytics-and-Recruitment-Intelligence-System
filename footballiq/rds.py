"""Minimal reader for R .rds files holding data frames (XDR serialisation, versions 2 and 3)."""
import bz2
import gzip
import lzma
import struct

import numpy as np
import pandas as pd

NA_INT = -2147483648


class _Reader:
    def __init__(self, data: bytes):
        self.b, self.p, self.refs = data, 0, []

    def int(self):
        v = struct.unpack_from(">i", self.b, self.p)[0]
        self.p += 4
        return v

    def length(self):
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def attrs(self, has_attr):
        if not has_attr:
            return {}
        out, node = {}, self.item()
        for tag, value in node or []:
            out[tag] = value
        return out

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == 254 or t in (253, 242, 241, 251, 250):
            return None
        if t == 255:
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if t == 1:                                   # symbol
            name = self.item()
            self.refs.append(name)
            return name
        if t in (249, 248, 247):                     # package / persistent / namespace
            self.int()
            names = [self.item() for _ in range(self.int())]
            self.refs.append(names)
            return names
        if t in (2, 6, 5, 17):                       # pairlist-like: flatten into [(tag, value)]
            out = []
            while True:
                if has_attr:
                    self.item()
                tag = self.item() if has_tag else None
                out.append((tag, self.item()))
                flags = self.int()
                t2 = flags & 0xFF
                if t2 == 254:
                    return out
                if t2 not in (2, 6, 5, 17):
                    self.p -= 4
                    out.append((None, self.item()))
                    return out
                has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == 9:                                   # CHARSXP
            n = self.int()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n]
            self.p += n
            try:
                return s.decode("utf-8")
            except UnicodeDecodeError:
                return s.decode("latin-1")
        if t in (10, 13):
            n = self.length()
            arr = np.frombuffer(self.b, dtype=">i4", count=n, offset=self.p).astype(np.int64)
            self.p += 4 * n
            return {"type": "lgl" if t == 10 else "int", "data": arr, "attr": self.attrs(has_attr)}
        if t == 14:
            n = self.length()
            arr = np.frombuffer(self.b, dtype=">f8", count=n, offset=self.p).astype(float)
            self.p += 8 * n
            return {"type": "real", "data": arr, "attr": self.attrs(has_attr)}
        if t == 16:
            n = self.length()
            data = [self.item() for _ in range(n)]
            return {"type": "str", "data": data, "attr": self.attrs(has_attr)}
        if t in (19, 20):
            n = self.length()
            data = [self.item() for _ in range(n)]
            return {"type": "list", "data": data, "attr": self.attrs(has_attr)}
        if t == 24:
            n = self.length()
            self.p += n
            return {"type": "raw", "data": None, "attr": self.attrs(has_attr)}
        if t == 238:                                 # ALTREP
            info, state, attr = self.item(), self.item(), self.item()
            cls = info[0][1]
            attr = dict(attr) if attr else {}
            if cls in ("compact_intseq", "compact_realseq"):
                n, start, step = state["data"]
                data = start + step * np.arange(int(n))
                return {"type": "int" if cls == "compact_intseq" else "real", "data": data, "attr": attr}
            if cls.startswith("wrap_"):
                inner = state["data"][0] if isinstance(state, dict) else state[0][1]
                return {**inner, "attr": {**inner.get("attr", {}), **attr}}
            if cls == "deferred_string":
                inner = state[0][1]
                data = [None if (isinstance(v, float) and np.isnan(v)) or v == NA_INT else
                        (str(int(v)) if float(v).is_integer() else repr(float(v))) for v in inner["data"]]
                return {"type": "str", "data": data, "attr": attr}
            raise ValueError(f"unsupported ALTREP class {cls}")
        raise ValueError(f"unsupported SEXP type {t} at byte {self.p}")


def _column(obj):
    kind, data, attr = obj["type"], obj["data"], obj.get("attr", {})
    if kind == "str":
        return pd.array(data, dtype=object)
    if kind == "real":
        return data
    if "levels" in attr:                              # factor
        levels = attr["levels"]["data"]
        return pd.array([None if v == NA_INT else levels[v - 1] for v in data], dtype=object)
    out = data.astype(float)
    out[data == NA_INT] = np.nan
    if kind == "lgl" or not np.isnan(out).any():
        return data if not (data == NA_INT).any() else out
    return out


def read_rds(path) -> pd.DataFrame:
    raw = open(path, "rb").read()
    for opener in (gzip.decompress, bz2.decompress, lzma.decompress, lambda b: b):
        try:
            data = opener(raw)
            break
        except Exception:
            continue
    if data[:2] != b"X\n":
        raise ValueError("not an XDR-serialised RDS file")
    r = _Reader(data)
    r.p = 2
    version = r.int()
    r.int(); r.int()
    if version >= 3:
        n = r.int()
        r.p += n
    obj = r.item()
    names = obj["attr"]["names"]["data"]
    return pd.DataFrame({name: _column(col) for name, col in zip(names, obj["data"])})
