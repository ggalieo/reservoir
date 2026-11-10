"""分块并发下载 PyTorch 轮子：32 MB 一块，8 并发，逐块校验。

大范围 Range 请求会被镜像挂住，小分块则稳定；失败只重试单块。
"""
import concurrent.futures as cf
import os
import ssl
import sys
import threading
import time
import urllib.request

URL = ("https://mirrors.aliyun.com/pytorch-wheels/cu124/"
       "torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl")
PROJ = r"C:\油气藏多源数据表征\reservoir-repr"
OUT = os.path.join(PROJ, "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl")
TMP = os.path.join(PROJ, "_dl")
CHUNK = 32 * 1024 * 1024
WORKERS = 8

CTX = ssl.create_default_context()
lock = threading.Lock()
done_bytes = 0


def total_size():
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        cr = r.headers.get("Content-Range") or ""
    return int(cr.split("/")[-1])


def get_chunk(idx, start, end, total):
    global done_bytes
    path = os.path.join(TMP, f"c{idx:04d}")
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        with lock:
            done_bytes += want
        return None
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}")
                got = 0
                with open(path, "wb") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        got += len(buf)
            if os.path.getsize(path) != want:
                raise RuntimeError(f"size {os.path.getsize(path)} != {want}")
            with lock:
                done_bytes += want
                pct = done_bytes / total * 100
                print(f"  [{pct:5.1f}%] chunk {idx} ok", flush=True)
            return None
        except Exception as e:
            if attempt == 4:
                return f"chunk {idx} 失败: {type(e).__name__} {e}"
            time.sleep(1.5 * (attempt + 1))
    return f"chunk {idx} 失败"


def main():
    os.makedirs(TMP, exist_ok=True)
    total = total_size()
    print(f"total {total/1e6:.1f} MB, chunk {CHUNK/1e6:.0f} MB, workers {WORKERS}", flush=True)

    spans = []
    idx = 0
    s = 0
    while s < total:
        e = min(s + CHUNK - 1, total - 1)
        spans.append((idx, s, e))
        idx += 1
        s = e + 1
    print(f"{len(spans)} chunks", flush=True)

    t0 = time.time()
    errs = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(get_chunk, i, s, e, total) for i, s, e in spans]
        for f in cf.as_completed(futs):
            r = f.result()
            if r:
                errs.append(r)
                print("  !!", r, flush=True)

    if errs:
        sys.exit(f"\n{len(errs)} 块失败，重跑本脚本会跳过已完成的块\n" + "\n".join(errs[:5]))

    print("\n拼接中...", flush=True)
    with open(OUT, "wb") as out:
        for i, _, _ in spans:
            with open(os.path.join(TMP, f"c{i:04d}"), "rb") as f:
                while True:
                    buf = f.read(1 << 23)
                    if not buf:
                        break
                    out.write(buf)

    size = os.path.getsize(OUT)
    dt = time.time() - t0
    print(f"完成 {size/1e6:.1f} MB / 期望 {total/1e6:.1f} MB，耗时 {dt/60:.1f} 分钟，"
          f"平均 {total/1e6/dt:.2f} MB/s", flush=True)
    if size != total:
        sys.exit("大小校验失败")
    print("校验通过", flush=True)


if __name__ == "__main__":
    main()
