#!/usr/bin/env python3
"""一次性验证服务（compose 服务名: verify）。

按顺序执行三个阶段，并以退出码汇报总体结果：

1. 包构建：用 setuptools 构建 wheel 到临时目录；
2. 代码测试：运行仓库内全部 unittest 测试；
3. HTTP 冒烟：对运行中的 API 提交
   - 一条可行轨迹（期望 feasible=true、漂移/区间/残差证据齐全），
   - 一条无解轨迹（期望 feasible=false 且给出明确结论），
   - 一条非法请求（期望 HTTP 400），
   - 一条带缺口占位的可行轨迹（期望占位观测被标记且不参与残差），
   - 一条只有让某级完全由无效采样支撑才能对齐的轨迹
     （期望 feasible=false 且给出明确结论），
   - 三条非法 dropout_runs（越界 / 重叠 / 总数超限，期望 HTTP 400）。

API 地址取环境变量 ``API_BASE_URL``（compose 中为 http://api:8000）；
若该地址不可达且未显式要求使用远端服务，则在本地以随机端口临时启动
一个服务实例进行冒烟，方便脱离 compose 直接运行本脚本。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "/api/current-traces/align"

FEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [12, 19, 31, 38, 52, 58, 71, 79],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}

INFEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [900, 901, 902, 903, 904, 905, 906, 907],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}

INVALID_CASE: Dict[str, Any] = {
    "reference_levels": [1, 2, 3],  # 少于 8 个
    "observations": [1, 2, 3, 4, 5, 6, 7, 8],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 0,
}

# 缺口场景：两个 dropout 区间，占位值 999 若参与残差必超限（limit=0）。
DROPOUT_FEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [10, 999, 20, 30, 999, 40, 50, 60, 70, 80],
    "drift_min": -2,
    "drift_max": 2,
    "residual_limit": 0,
    "dwell_min": 1,
    "dwell_max": 3,
    "dropout_runs": [{"start": 1, "end": 2}, {"start": 4, "end": 5}],
}

# 有效样本仅 5 个 < 最少 6 级：只有让某级完全由无效采样支撑才能对齐。
DROPOUT_INFEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [10, 20, 999, 999, 999, 60, 70, 80],
    "drift_min": -2,
    "drift_max": 2,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 3,
    "dropout_runs": [{"start": 2, "end": 5}],
}

# 非法 dropout_runs：越界 / 重叠 / 无效采样总数超限。
DROPOUT_INVALID_CASES: Dict[str, Dict[str, Any]] = {
    "out_of_range": {"dropout_runs": [{"start": 7, "end": 11}]},
    "overlap": {
        "dropout_runs": [{"start": 1, "end": 3}, {"start": 2, "end": 4}]
    },
    "too_many_samples": {
        "dropout_runs": [{"start": 0, "end": 4}, {"start": 4, "end": 7}]
    },
}


class StageFailure(Exception):
    pass


def _log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def stage_build_package() -> Path:
    _log("build", "开始构建 wheel ...")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            tmp,
            str(REPO_ROOT),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise StageFailure("wheel 构建失败")
        wheels = list(Path(tmp).glob("*.whl"))
        if not wheels:
            raise StageFailure("未找到构建产物 wheel")
        # wheel 位于临时目录，复制到持久临时路径以便汇报。
        out_dir = Path(tempfile.mkdtemp(prefix="nanopore-wheel-"))
        wheel = out_dir / wheels[0].name
        wheel.write_bytes(wheels[0].read_bytes())
    _log("build", f"包构建成功: {wheel.name}")
    return wheel


def stage_run_tests() -> None:
    _log("tests", "运行 unittest 测试套件 ...")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise StageFailure("代码测试失败")
    _log("tests", "全部测试通过")


def _http_request(
    base_url: str, payload: Any
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_healthy(base_url: str, attempts: int = 30) -> None:
    url = base_url.rstrip("/") + "/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise StageFailure(f"服务健康检查未通过: {url}")


def _maybe_start_local_server() -> Tuple[Optional[subprocess.Popen], str]:
    """优先使用 API_BASE_URL；不可达时本地起随机端口服务。"""
    base = os.environ.get("API_BASE_URL")
    if base:
        _wait_healthy(base)
        return None, base

    _log("smoke", "API_BASE_URL 未设置，本地临时启动 API 实例 ...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nanopore_align.app",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    port: Optional[int] = None
    deadline = time.time() + 10
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise StageFailure("本地 API 进程提前退出")
            time.sleep(0.05)
            continue
        if line.startswith("LISTENING "):
            port = int(line.split()[1])
            break
    if port is None:
        proc.kill()
        raise StageFailure("未能读取本地 API 端口")
    base = f"http://127.0.0.1:{port}"
    _wait_healthy(base)
    return proc, base


def stage_http_smoke(base_url: str) -> None:
    _log("smoke", f"对 {base_url} 发起 HTTP 冒烟 ...")

    status, body = _http_request(base_url, FEASIBLE_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(f"可行轨迹用例失败: status={status} body={body}")
    if not isinstance(body.get("drift"), int):
        raise StageFailure("可行轨迹缺少整数漂移字段")
    levels = body.get("levels")
    if not isinstance(levels, list) or not levels:
        raise StageFailure("可行轨迹缺少逐级采样区间")
    covered = [s["index"] for lv in levels for s in lv["samples"]]
    if covered != list(range(len(FEASIBLE_CASE["observations"]))):
        raise StageFailure("残差证据未恰好覆盖每个观测一次")
    _log(
        "smoke",
        f"可行轨迹通过: drift={body['drift']} "
        f"levels={len(levels)} sum={body['residual_sum']} "
        f"max={body['max_abs_residual']}",
    )

    status, body = _http_request(base_url, INFEASIBLE_CASE)
    if status != 200 or body.get("feasible") is not False:
        raise StageFailure(f"无解轨迹用例失败: status={status} body={body}")
    if body.get("reason") != "no_alignment_exists":
        raise StageFailure("无解轨迹缺少明确结论 reason")
    _log("smoke", "无解轨迹通过: 服务返回 feasible=false 及明确结论")

    status, body = _http_request(base_url, INVALID_CASE)
    if status != 400:
        raise StageFailure(f"非法请求应返回 400，实际 status={status}")
    _log("smoke", "非法请求通过: 返回 HTTP 400 拒绝")

    # 缺口场景：占位观测被标记、归属单一电平且不参与残差。
    status, body = _http_request(base_url, DROPOUT_FEASIBLE_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(
            f"缺口可行轨迹用例失败: status={status} body={body}"
        )
    if body.get("residual_sum") != 0 or body.get("max_abs_residual") != 0:
        raise StageFailure("缺口占位观测参与了残差统计")
    levels = body.get("levels") or []
    covered = [s["index"] for lv in levels for s in lv["samples"]]
    if covered != list(range(len(DROPOUT_FEASIBLE_CASE["observations"]))):
        raise StageFailure("缺口轨迹未恰好覆盖每个观测一次")
    dropout_hits = [
        (lv, s)
        for lv in levels
        for s in lv["samples"]
        if s.get("dropout")
    ]
    if sorted(s["index"] for _, s in dropout_hits) != [1, 4]:
        raise StageFailure("缺口样本未在逐样本证据中逐一标明")
    for lv, s in dropout_hits:
        if s.get("residual") is not None:
            raise StageFailure("缺口样本不应携带残差值")
        if not (lv["sample_start"] <= s["index"] < lv["sample_end"]):
            raise StageFailure("缺口样本未归属其采用电平的采样区间")
    _log(
        "smoke",
        f"缺口可行轨迹通过: drift={body['drift']} "
        f"dropouts={[s['index'] for _, s in dropout_hits]} 已标记且不计残差",
    )

    status, body = _http_request(base_url, DROPOUT_INFEASIBLE_CASE)
    if status != 200 or body.get("feasible") is not False:
        raise StageFailure(
            f"缺口无解轨迹用例失败: status={status} body={body}"
        )
    if body.get("reason") != "no_alignment_exists":
        raise StageFailure("缺口无解轨迹缺少明确结论 reason")
    _log("smoke", "缺口无解轨迹通过: 全无效级支撑被拒绝并给出明确结论")

    for name, patch in DROPOUT_INVALID_CASES.items():
        payload = dict(DROPOUT_FEASIBLE_CASE)
        payload.update(patch)
        status, body = _http_request(base_url, payload)
        if status != 400:
            raise StageFailure(
                f"非法 dropout_runs（{name}）应返回 400，实际 status={status}"
            )
    _log("smoke", "非法 dropout_runs 通过: 越界/重叠/总数超限均返回 400")


def main() -> int:
    failures = []
    wheel: Optional[Path] = None
    proc: Optional[subprocess.Popen] = None
    try:
        wheel = stage_build_package()
    except StageFailure as exc:
        failures.append(f"包构建: {exc}")

    try:
        stage_run_tests()
    except StageFailure as exc:
        failures.append(f"代码测试: {exc}")

    try:
        proc, base_url = _maybe_start_local_server()
        stage_http_smoke(base_url)
    except StageFailure as exc:
        failures.append(f"HTTP 冒烟: {exc}")

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("-" * 60)
    if failures:
        print("VERIFY 结果: 失败")
        for item in failures:
            print(f"  - {item}")
        if wheel is not None:
            print(f"  wheel 产物: {wheel}")
        return 1

    print("VERIFY 结果: 全部通过（包构建 / 代码测试 / HTTP 冒烟）")
    if wheel is not None:
        print(f"  wheel 产物: {wheel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
