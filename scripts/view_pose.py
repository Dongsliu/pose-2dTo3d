#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在浏览器里预览姿势，并可从页面发起检测。"""

import argparse
import base64
import json
import re
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.image_io import imread  # noqa: E402
from core.pipeline import detect_image  # noqa: E402

viewer_src = project_root / 'viewer' / 'index.html'
UPLOAD_DIR = project_root / 'viewer' / '_uploads'
SAMPLES_DIR = project_root / 'samples' / 'poses'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
EXPORT_DIR = project_root / 'viewer' / '_exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
export_lock = threading.Lock()
MAX_REQUEST_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def image_to_data_url(path: Path) -> str:
    image = imread(path)
    if image is None:
        raise RuntimeError(f'无法读取图片: {path}')
    ok, buf = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('图片编码失败')
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def ndarray_to_data_url(image) -> str:
    ok, buf = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('图片编码失败')
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def decode_data_url(data_url: str):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('无法解码上传图片')
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise RuntimeError('Uploaded image is too large')
    return image


def windows_to_wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', resolved)
    if not match:
        raise RuntimeError(f'Path is not on a Windows drive: {resolved}')
    return f'/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), "/")}'


def export_pose_fbx(pose, output_path: Path):
    blender_script = project_root / 'scripts' / 'export_pose_fbx.py'
    template_path = project_root / 'product' / 'PoseRig.blend'
    if not template_path.exists():
        raise RuntimeError(f'Missing FBX template: {template_path}')
    if not blender_script.exists():
        raise RuntimeError(f'Missing Blender exporter: {blender_script}')

    pose_path = output_path.with_suffix('.pose.json')
    pose_path.write_text(json.dumps(pose, ensure_ascii=False), encoding='utf-8')
    configured_blender = os.environ.get('POSE_BLENDER_EXE', '').strip()
    blender_candidates = [
        Path(configured_blender) if configured_blender else None,
        Path(r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'),
        Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'),
    ]
    blender_exe = next((path for path in blender_candidates if path and path.exists()), None)
    if blender_exe is None:
        blender_exe = shutil.which('blender.exe') or shutil.which('blender')

    if blender_exe:
        command = [
            str(blender_exe), '--background', str(template_path),
            '--python', str(blender_script), '--',
            '--pose', str(pose_path),
            '--template', str(template_path),
            '--output', str(output_path),
        ]
    else:
        command = [
            'wsl.exe', '-e', 'blender', '--background', windows_to_wsl_path(template_path),
            '--python', windows_to_wsl_path(blender_script), '--',
            '--pose', windows_to_wsl_path(pose_path),
            '--template', windows_to_wsl_path(template_path),
            '--output', windows_to_wsl_path(output_path),
        ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        pose_path.unlink(missing_ok=True)
    if completed.returncode != 0 or not output_path.exists():
        details = (completed.stderr or completed.stdout or '').strip()
        if len(details) > 2000:
            details = details[-2000:]
        raise RuntimeError(f'Blender FBX export failed ({completed.returncode}): {details}')


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(args[0] if args else format)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            html = viewer_src.read_text(encoding='utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            return
        if parsed.path == '/api/samples':
            items = []
            for path in sorted(SAMPLES_DIR.iterdir(), key=lambda p: p.name.lower()):
                if path.suffix.lower() in IMAGE_EXTS:
                    items.append({'name': path.stem, 'file': path.name})
            self._json(200, {'ok': True, 'items': items})
            return
        if parsed.path.startswith('/samples/'):
            name = Path(unquote(parsed.path[len('/samples/'):])).name
            path = SAMPLES_DIR / name
            if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
                self.send_error(404)
                return
            mime = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp',
            }.get(path.suffix.lower(), 'application/octet-stream')
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ('/api/detect', '/api/export-fbx'):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send_error(400, 'Invalid Content-Length')
            return
        if length <= 0:
            self.send_error(400, 'Request body is required')
            return
        if length > MAX_REQUEST_BYTES:
            self.send_error(413, 'Request body is too large')
            return
        try:
            body = json.loads(self.rfile.read(length).decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, 'Invalid JSON request')
            return
        if parsed.path == '/api/export-fbx':
            try:
                pose = body.get('pose')
                if not pose or not pose.get('keypoints_3d'):
                    raise RuntimeError('No detected pose was provided')
                source_name = Path(body.get('filename') or 'pose').stem
                safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', source_name).strip('._') or 'pose'
                output_path = EXPORT_DIR / f'{safe_name}_pose.fbx'
                with export_lock:
                    export_pose_fbx(pose, output_path)
                    payload = output_path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', f'attachment; filename="{safe_name}_pose.fbx"')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                output_path.unlink(missing_ok=True)
            except Exception as exc:
                self._json(500, {'ok': False, 'error': str(exc)})
            return
        try:
            image = decode_data_url(body.get('image') or '')
            name = Path(body.get('filename') or 'upload.jpg').name
            save_path = UPLOAD_DIR / name
            ext = save_path.suffix.lower() if save_path.suffix else '.jpg'
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                ext = '.jpg'
            ok, buf = cv2.imencode(ext, image)
            if not ok:
                raise RuntimeError('保存上传图片失败')
            buf.tofile(str(save_path))
            pose = detect_image(
                image,
                model_name=body.get('model') or 'rtmw-x',
                device=body.get('device') or 'auto',
                refine_hands=bool(body.get('refine_hands')),
                constraint_strength=float(body.get('constraint_strength') or 0.5),
                input_image=str(save_path),
            )
            out_json = project_root / 'pose.json'
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(pose, f, indent=2, ensure_ascii=False)
            payload = {
                'ok': True,
                'pose': pose,
                'image': ndarray_to_data_url(image),
            }
            raw = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(raw)
        except Exception as exc:
            raw = json.dumps({'ok': False, 'error': str(exc)}).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(raw)


def local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith('127.'):
                ips.append(ip)
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('8.8.8.8', 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip not in ips and not ip.startswith('127.'):
            ips.insert(0, ip)
    except Exception:
        pass
    return ips


def main():
    parser = argparse.ArgumentParser(description='浏览器预览姿势检测结果')
    parser.add_argument('pose', nargs='?', default='', help='可选：启动时加载的 pose.json')
    parser.add_argument('--image', help='可选原图')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址，默认仅本机访问')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    print('首次检测会加载模型，可能需要几十秒。之后同一进程会快很多。')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f'本机: http://127.0.0.1:{args.port}/')
    for ip in local_ips():
        print(f'局域网: http://{ip}:{args.port}/')
    print('拖入图片后点「开始检测」。关闭这个窗口即停止服务。')
    webbrowser.open(f'http://127.0.0.1:{args.port}/')
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
