"""HTTP 层测试：直接调用处理函数与真实 HTTP 端到端冒烟。"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, Dict, Tuple

from nanopore_align.app import align_from_payload, build_server


FEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [15, 25, 35, 45, 55, 65, 75, 85],
    "drift_min": -10,
    "drift_max": 10,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}

INFEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [900] * 8,
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}

DROPOUT_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [10, 20, 999, 30, 40, 50, 60, 70, 80],
    "drift_min": -2,
    "drift_max": 2,
    "residual_limit": 0,
    "dwell_min": 1,
    "dwell_max": 3,
    "dropout_runs": [{"start": 2, "end": 3}],
}


class AlignFromPayloadTests(unittest.TestCase):
    def test_feasible(self) -> None:
        status, body = align_from_payload(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertEqual(body["drift"], 5)
        self.assertEqual(body["residual_sum"], 0)
        self.assertEqual(len(body["levels"]), 8)

    def test_infeasible_is_explicit_conclusion(self) -> None:
        status, body = align_from_payload(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_exists")
        self.assertIn("message", body)

    def test_missing_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        del bad["drift_max"]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["error"], "missing_fields")

    def test_unknown_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["extra"] = 1
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unknown_fields")

    def test_bad_size_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["reference_levels"] = list(range(7))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_too_many_observations_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["observations"] = list(range(61))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)

    def test_wrong_type_rejected(self) -> None:
        status, body = align_from_payload("not-a-dict")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_body")


class DropoutPayloadTests(unittest.TestCase):
    """dropout_runs 字段：接收、转换、求解与字段级拒绝。"""

    def test_feasible_with_dropout(self) -> None:
        status, body = align_from_payload(DROPOUT_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertEqual(body["residual_sum"], 0)
        drops = [
            (lv["level_order"], s)
            for lv in body["levels"]
            for s in lv["samples"]
            if s.get("dropout")
        ]
        self.assertEqual(len(drops), 1)
        level_order, sample = drops[0]
        self.assertEqual(sample["index"], 2)
        self.assertIsNone(sample["residual"])
        # 归属：缺口样本落在其采用电平的采样区间内。
        lv = body["levels"][level_order]
        self.assertLessEqual(lv["sample_start"], 2)
        self.assertLess(2, lv["sample_end"])

    def test_dropout_infeasible_is_explicit_conclusion(self) -> None:
        # 有效样本不足以让每个采用电平至少含一个 -> 既有无解结论。
        bad = dict(DROPOUT_PAYLOAD)
        bad["observations"] = [10, 20, 999, 999, 999, 60, 70, 80]
        bad["residual_limit"] = 2
        bad["dropout_runs"] = [{"start": 2, "end": 5}]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_exists")

    def test_overlap_rejected(self) -> None:
        bad = dict(DROPOUT_PAYLOAD)
        bad["dropout_runs"] = [{"start": 1, "end": 3}, {"start": 2, "end": 4}]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_out_of_range_rejected(self) -> None:
        bad = dict(DROPOUT_PAYLOAD)
        bad["dropout_runs"] = [{"start": 6, "end": 9}]  # N=9 时 end=9 合法，
        status, _ = align_from_payload(bad)  # 恰好覆盖末样本，允许。
        self.assertEqual(status, 200)
        bad["dropout_runs"] = [{"start": 6, "end": 10}]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")
        bad["dropout_runs"] = [{"start": -1, "end": 2}]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)

    def test_too_many_runs_rejected(self) -> None:
        bad = dict(DROPOUT_PAYLOAD)
        bad["dropout_runs"] = [
            {"start": 0, "end": 1},
            {"start": 2, "end": 3},
            {"start": 4, "end": 5},
            {"start": 6, "end": 7},
        ]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_too_many_dropout_samples_rejected(self) -> None:
        bad = dict(DROPOUT_PAYLOAD)
        bad["dropout_runs"] = [{"start": 0, "end": 4}, {"start": 4, "end": 7}]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_malformed_run_items_rejected(self) -> None:
        for runs in (
            "not-a-list",
            [{"start": 0}],
            [{"start": 0, "end": 1, "extra": 2}],
            [[0, 1]],
            [1],
        ):
            bad = dict(DROPOUT_PAYLOAD)
            bad["dropout_runs"] = runs
            status, body = align_from_payload(bad)
            self.assertEqual(status, 400, msg=f"dropout_runs={runs}")
            self.assertEqual(body["error"], "invalid_request")


class HttpEndToEndTests(unittest.TestCase):
    server: ThreadingHTTPServer
    thread: threading.Thread
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = build_server("127.0.0.1", 0)
        port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _post(self, payload: Any) -> Tuple[int, Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self) -> None:
        with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["status"], "ok")

    def test_feasible_roundtrip(self) -> None:
        status, body = self._post(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        covered = [
            s["index"]
            for lv in body["levels"]
            for s in lv["samples"]
        ]
        self.assertEqual(covered, list(range(8)))

    def test_infeasible_roundtrip(self) -> None:
        status, body = self._post(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])

    def test_invalid_roundtrip(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["drift_min"] = 99
        bad["drift_max"] = 0
        status, body = self._post(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])

    def test_malformed_json(self) -> None:
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("应当返回 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_unknown_route_404(self) -> None:
        try:
            urllib.request.urlopen(f"{self.base}/nope", timeout=5)
            self.fail("应当返回 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)


if __name__ == "__main__":
    unittest.main()
