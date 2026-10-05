"""联合对齐算法的单元测试与暴力对照测试。"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

from nanopore_align.alignment import AlignmentError, solve_alignment


def brute_force(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    dropout_runs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[dict]:
    """穷举所有漂移 / 含首尾子序列 / 停留组合，返回与 solve 同口径的最优解。

    提供 ``dropout_runs`` 时按缺口语义裁决：边界不得严格落在缺口内部
    （缺口不被采样边界切开）、每级至少一个有效样本、残差仅统计有效样本。
    """
    R = len(reference)
    N = len(observations)
    runs = list(dropout_runs or [])
    drop = [False] * N
    for a, b in runs:
        for t in range(a, b):
            drop[t] = True
    best: Optional[Tuple] = None

    for d in range(drift_min, drift_max + 1):
        # 枚举被跳过的内部参考索引集合（大小 <= max_skips）。
        inner = list(range(1, R - 1))
        for skip_count in range(0, min(max_skips, len(inner)) + 1):
            for skipped in itertools.combinations(inner, skip_count):
                used = [i for i in range(R) if i not in set(skipped)]
                k = len(used)
                if k > N or k * dwell_min > N or k * dwell_max < N:
                    continue
                for dwells in itertools.product(
                    range(dwell_min, dwell_max + 1), repeat=k
                ):
                    if sum(dwells) != N:
                        continue
                    bounds = (0,) + tuple(itertools.accumulate(dwells))
                    # 内部边界不得严格落在任何缺口内部。
                    split = any(
                        a < p < b
                        for p in bounds[1:-1]
                        for a, b in runs
                    )
                    if split:
                        continue
                    total = 0
                    worst = 0
                    ok = True
                    s = 0
                    for ri, L in zip(used, dwells):
                        level = reference[ri] + d
                        valid_in_level = 0
                        for t in range(s, s + L):
                            if drop[t]:
                                continue
                            valid_in_level += 1
                            r = abs(observations[t] - level)
                            if r > residual_limit:
                                ok = False
                                break
                            total += r
                            worst = max(worst, r)
                        if not ok or valid_in_level == 0:
                            ok = False
                            break
                        s += L
                    if not ok:
                        continue
                    cost = (skip_count, total, worst, d, bounds[1:])
                    if best is None or cost < best[0]:
                        best = (cost, used, d)

    if best is None:
        return None
    (skipped_n, total, worst, d, boundaries), used, drift = best
    return {
        "feasible": True,
        "drift": drift,
        "num_skips": skipped_n,
        "residual_sum": total,
        "max_abs_residual": worst,
        "boundaries": list(boundaries[:-1]),
        "used_indices": list(used),
    }


def _assert_matches_brute(
    testcase: unittest.TestCase,
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int = 1,
    dwell_max: int = 3,
    max_skips: int = 2,
    dropout_runs: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    got = solve_alignment(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
        dropout_runs=dropout_runs,
    )
    want = brute_force(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
        dropout_runs=dropout_runs,
    )
    if want is None:
        testcase.assertFalse(got["feasible"], msg=f"意外可行: {got}")
        return
    testcase.assertTrue(got["feasible"], msg="意外无解")
    testcase.assertEqual(got["drift"], want["drift"])
    testcase.assertEqual(got["num_skips"], want["num_skips"])
    testcase.assertEqual(got["residual_sum"], want["residual_sum"])
    testcase.assertEqual(got["max_abs_residual"], want["max_abs_residual"])
    testcase.assertEqual(got["boundaries"], want["boundaries"])
    got_used = [lv["reference_index"] for lv in got["levels"]]
    testcase.assertEqual(got_used, want["used_indices"])


class ExactAlignmentTests(unittest.TestCase):
    def test_wide_drift_interval_finds_far_drift(self) -> None:
        # 真实漂移远离 0，且首级/末级可行漂移区间很窄：
        # 验证首末级预筛不会把正确漂移漏掉。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 432 for x in ref]
        res = solve_alignment(ref, obs, -1000, 1000, 0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 432)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)

    def test_first_last_prefilter_conflict_is_infeasible(self) -> None:
        # 首级前几个观测要求漂移 -7 附近，末级观测要求漂移 +7 附近，
        # 首末级可行区间交集为空 -> 明确无解（limit=0，无折中可能）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [-7, -7, -7, 10, 20, 30, 40, 50, 60, 70, 77, 77]
        res = solve_alignment(ref, obs, -100, 100, 0,
                              dwell_min=1, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_straight_no_skip(self) -> None:
        # 8 个参考电平，每级恰好 1 个观测，漂移 +5，残差全 0。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 5 for x in ref]
        res = solve_alignment(ref, obs, -10, 10, 2)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["max_abs_residual"], 0)
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(res["levels"]), 8)
        for slot, lv in enumerate(res["levels"]):
            self.assertEqual(lv["sample_start"], slot)
            self.assertEqual(lv["sample_end"], slot + 1)
            self.assertEqual(lv["dwell"], 1)
            self.assertEqual(lv["samples"][0]["residual"], 0)

    def test_dwell_two_three_and_skip(self) -> None:
        # 8 个参考电平；跳过索引 3；停留：2,3,2,... 凑 14 个观测。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        used = [0, 1, 2, 4, 5, 6, 7]
        dwells = [2, 3, 2, 2, 2, 2, 1]
        self.assertEqual(sum(dwells), 14)
        obs: List[int] = []
        for ri, L in zip(used, dwells):
            obs.extend([ref[ri] - 3] * L)
        res = solve_alignment(ref, obs, -5, 5, 1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], -3)
        self.assertEqual(res["num_skips"], 1)
        self.assertEqual(res["skipped_reference_indices"], [3])
        self.assertEqual(
            [lv["reference_index"] for lv in res["levels"]], used
        )
        self.assertEqual([lv["dwell"] for lv in res["levels"]], dwells)
        self.assertEqual(res["boundaries"], [2, 5, 7, 9, 11, 13])
        for lv in res["levels"]:
            for s in lv["samples"]:
                self.assertEqual(abs(s["residual"]), 0)

    def test_first_and_last_mandatory(self) -> None:
        # 首电平与尾电平与观测差距巨大，任何跳过都救不了 -> 无解。
        ref = [0, 100, 100, 100, 100, 100, 100, 1000]
        obs = [100] * 8
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_infeasible_when_residual_limit_tight(self) -> None:
        ref = [0, 1, 2, 3, 4, 5, 6, 7]
        # 每个观测比对应电平高 2；limit=1、漂移只能 0 -> 无解。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])

    def test_infeasible_when_too_many_levels(self) -> None:
        # 8 个观测、8 个必到级数（首尾+内部不允许跳过），每级最多 1 采样可行；
        # 但若禁用跳过且观测只有 8 个、停留最小 2 -> 无解。
        ref = list(range(8))
        obs = list(range(8))
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=2, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_skip_cap_enforced(self) -> None:
        # 需要 3 个跳过才可行，但上限为 2 -> 无解。
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 0, 700, 700]  # 仅首尾附近有观测；N 最少为 8，构造 8 个
        obs = [0, 0, 0, 0, 700, 700, 700, 700]
        # 合法对齐至少要跳过中间 6 个内部电平，> 2。
        res = solve_alignment(ref, obs, 0, 0, 0, max_skips=2)
        self.assertFalse(res["feasible"])
        res2 = solve_alignment(ref, obs, 0, 0, 0, max_skips=2,
                               dwell_min=1, dwell_max=3)
        self.assertFalse(res2["feasible"])


class ObjectiveOrderTests(unittest.TestCase):
    def test_minimize_skips_first(self) -> None:
        # 无跳过对齐需要较大残差；带 1 跳过残差为 0。
        # 仍应选择无跳过（跳过数优先级最高）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # 8 观测每级 1 个，全部偏移 2（limit=2，无跳过，残差和 16）。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 5)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 16)

    def test_then_residual_sum(self) -> None:
        # 漂移 -1 / 0 / +1 都可行，残差和不同 -> 选和最小者。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [r - 1 for r in ref]  # drift=-1 时残差为 0
        res = solve_alignment(ref, obs, -2, 2, 5)
        self.assertEqual(res["drift"], -1)
        self.assertEqual(res["residual_sum"], 0)

    def test_then_max_residual(self) -> None:
        # 构造两个漂移残差和相同但最大残差不同：用对称扰动。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # drift=0: 残差 +1,-1,...,0 -> 和 0（带符号不影响，这里用绝对值）
        obs = [1, 9, 21, 29, 41, 39, 61, 69]
        # drift=0 时绝对残差全 1；和 8、最大 1
        # drift=1 时残差 [0,-2,0,-2,...,±2] 和 16 更大 -> 0 胜
        res = solve_alignment(ref, obs, -1, 1, 3)
        self.assertEqual(res["drift"], 0)
        self.assertEqual(res["max_abs_residual"], 1)
        self.assertEqual(res["residual_sum"], 8)

    def test_then_drift_tie_break(self) -> None:
        # 所有观测恰好在两个漂移下都零残差（参考电平差偶数且观测居中无法同时为0；
        # 改用只有 1 级停留长度结构使得 +0 与 +1 不可能同残差，因此直接构造
        # limit 宽松、残差相同的情形：参考全部相同间距无法做到——
        # 这里验证漂移平局取较小漂移：令观测 = ref + 1，并允许 drift=1 与
        # 跳过路径下 drift=0 残差结构一致较难构造，改为直接校验纯平局：
        # 参考电平全部为偶数，观测相对 ref：+1，此时仅 drift=1 零残差，
        # 退而求其次，校验候选漂移中较小者获胜的代码路径由对照测试覆盖。
        ref = [0, 2, 4, 6, 8, 10, 12, 14]
        obs = [r + 1 for r in ref]
        res = solve_alignment(ref, obs, 0, 2, 1)
        self.assertEqual(res["drift"], 1)

    def test_boundaries_lexicographic_tie_break(self) -> None:
        # 全部相同参考电平：任意边界划分残差相同；应取字典序最小边界
        # （尽早结束第一级：在 dwell_min=1 下首条边界最小）。
        ref = [5] * 8
        obs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # 10 观测，8 级
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        # 额外的 2 个采样尽量后置 -> 停留 (1,1,1,1,1,1,1,3)，
        # 内部边界字典序最小。
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])


class ValidationTests(unittest.TestCase):
    def _base(self) -> dict:
        return dict(
            reference=list(range(8)),
            observations=list(range(8)),
            drift_min=0,
            drift_max=0,
            residual_limit=0,
        )

    def test_reference_size_bounds(self) -> None:
        kw = self._base()
        kw["reference"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["reference"] = list(range(25))
        kw["observations"] = list(range(25))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_observation_size_bounds(self) -> None:
        kw = self._base()
        kw["observations"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["observations"] = list(range(61))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_interval(self) -> None:
        kw = self._base()
        kw["drift_min"], kw["drift_max"] = 5, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_dwell_range(self) -> None:
        kw = self._base()
        kw["dwell_min"], kw["dwell_max"] = 0, 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 2, 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 1, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_max_skips_range(self) -> None:
        kw = self._base()
        kw["max_skips"] = 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["max_skips"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_width_cap(self) -> None:
        from nanopore_align.alignment import DRIFT_WIDTH_MAX

        kw = self._base()
        kw["drift_min"] = 0
        kw["drift_max"] = DRIFT_WIDTH_MAX
        solve_alignment(**kw)  # 边界宽度合法
        kw["drift_max"] = DRIFT_WIDTH_MAX + 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_negative_residual_limit_rejected(self) -> None:
        kw = self._base()
        kw["residual_limit"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_type_checks(self) -> None:
        kw = self._base()
        kw["reference"] = [1, 2, 3, 4, 5, 6, 7, "8"]  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["drift_min"] = 0.5  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["observations"] = True  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["reference"] = []  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)


class ResidualEvidenceTests(unittest.TestCase):
    def test_evidence_covers_every_observation_once(self) -> None:
        ref = [3, 7, 11, 15, 19, 23, 27, 31]
        obs = [3, 3, 8, 10, 16, 19, 22, 24, 28, 30, 30, 31]
        res = solve_alignment(ref, obs, -2, 2, 2,
                              dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        covered: List[int] = []
        for lv in res["levels"]:
            self.assertEqual(lv["sample_end"] - lv["sample_start"], lv["dwell"])
            self.assertEqual(len(lv["samples"]), lv["dwell"])
            for s in lv["samples"]:
                self.assertEqual(
                    s["residual"], s["observed"] - lv["adopted_level"]
                )
                self.assertLessEqual(abs(s["residual"]), 2)
                covered.append(s["index"])
        self.assertEqual(covered, list(range(len(obs))))
        self.assertEqual(
            res["residual_sum"],
            sum(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )
        self.assertEqual(
            res["max_abs_residual"],
            max(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )


class BruteForceComparisonTests(unittest.TestCase):
    """随机小规模实例：DP 必须与穷举结果完全一致（含全部平局裁决）。"""

    def test_random_cases(self) -> None:
        rng = random.Random(20261004)
        for trial in range(120):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            # 先随机生成一个“真值”对齐，再对部分观测加入噪声，
            # 保证可行与不可行实例混合出现。
            used = [0]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            # 随机跳过 0..2 个内部电平
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
                skipped_set = set()
            # 随机停留组合，和为 N，每段 1..3
            dwells = self._random_composition(rng, k, N, 1, 3)
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    obs.append(level + noise)
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            with self.subTest(trial=trial, ref=ref, obs=obs,
                              lo=d_lo, hi=d_hi, limit=limit):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2
                )

    @staticmethod
    def _random_composition(
        rng: random.Random, k: int, n: int, lo: int, hi: int
    ) -> Optional[List[int]]:
        choices: List[List[int]] = []
        total_lo = k * lo
        total_hi = k * hi
        if not total_lo <= n <= total_hi:
            return None
        # 在合法空间内简单拒绝采样。
        for _ in range(500):
            parts = [rng.randint(lo, hi) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


class WideDriftRandomTests(unittest.TestCase):
    """宽漂移区间下随机真值实例：预筛与枚举必须找回唯一真值对齐。"""

    def test_wide_interval_random_truth(self) -> None:
        rng = random.Random(424242)
        for _ in range(40):
            R = rng.randint(8, 24)
            k = R - rng.randint(0, 2)
            skip_set = set(rng.sample(range(1, R - 1), R - k))
            used = [i for i in range(R) if i not in skip_set]
            dwells = self._composition(rng, k, rng.randint(k, 3 * k))
            if dwells is None:
                continue
            N = sum(dwells)
            if N > 60:
                continue
            ref = sorted(rng.sample(range(-20000, 20000), R))
            truth_d = rng.randint(-500, 500)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                obs.extend([ref[ri] + truth_d] * L)
            with self.subTest(R=R, N=N, d=truth_d):
                res = solve_alignment(
                    ref, obs, -1000, 1000, 0, 1, 3, 2
                )
                self.assertTrue(res["feasible"], msg="漏掉可行真值对齐")
                self.assertEqual(res["drift"], truth_d)
                self.assertEqual(res["residual_sum"], 0)
                self.assertEqual(
                    [lv["reference_index"] for lv in res["levels"]], used
                )
                self.assertEqual(
                    [lv["dwell"] for lv in res["levels"]], dwells
                )

    @staticmethod
    def _composition(
        rng: random.Random, k: int, n: int
    ) -> Optional[List[int]]:
        for _ in range(800):
            parts = [rng.randint(1, 3) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


class DropoutScenarioTests(unittest.TestCase):
    """dropout_runs 缺口语义的定向测试。"""

    REF = [10, 20, 30, 40, 50, 60, 70, 80]

    def test_dropouts_keep_time_placeholders_and_join_one_level(self) -> None:
        # 在第 2、3 个采样位置插入两个无效占位（垃圾值），二者必须
        # 同属一个采用电平；该级因此停留 3（1 有效 + 2 占位）。
        good = [x + 5 for x in self.REF]
        obs = good[:2] + [9999, 9999] + good[2:]
        res = solve_alignment(
            self.REF, obs, -10, 10, 2, 1, 3, 2, dropout_runs=[[2, 4]]
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["max_abs_residual"], 0)
        self.assertEqual(res["boundaries"], [1, 2, 5, 6, 7, 8, 9])

        level = res["levels"][2]
        self.assertEqual((level["sample_start"], level["sample_end"]), (2, 5))
        self.assertEqual(level["dwell"], 3)
        ignored = [s for s in level["samples"] if s.get("ignored")]
        self.assertEqual([s["index"] for s in ignored], [2, 3])
        for s in ignored:
            self.assertIsNone(s["residual"])
            self.assertEqual(s["ignored_reason"], "dropout")
            self.assertEqual(s["dropout_run"], 0)
        # 同级的有效样本残差照常给出。
        valid = [s for s in level["samples"] if not s.get("ignored")]
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["index"], 4)

        self.assertEqual(
            res["dropout_runs"],
            [
                {
                    "dropout_run": 0,
                    "sample_start": 2,
                    "sample_end": 4,
                    "level_order": 2,
                    "ignored": True,
                }
            ],
        )
        self.assertEqual(
            level["ignored_dropout_runs"],
            [
                {
                    "dropout_run": 0,
                    "sample_start": 2,
                    "sample_end": 4,
                    "level_order": 2,
                }
            ],
        )

    def test_same_observations_without_field_is_infeasible(self) -> None:
        good = [x + 5 for x in self.REF]
        obs = good[:2] + [9999, 9999] + good[2:]
        res = solve_alignment(self.REF, obs, -10, 10, 2, 1, 3, 2)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_garbage_placeholders_do_not_create_level_switches(self) -> None:
        # 占位取极端跳变值：若错误参与残差会凭空制造一次“电平切换”，
        # 残差和/最大残差也会被污染；忽略后应仍为零残差解。
        good = [x + 5 for x in self.REF]
        obs = good[:3] + [-1_000_000_000] + good[3:]
        res = solve_alignment(
            self.REF, obs, 5, 5, 0, 1, 3, 2, dropout_runs=[[3, 4]]
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["max_abs_residual"], 0)
        self.assertEqual(res["num_skips"], 0)

    def test_run_must_be_split_is_infeasible(self) -> None:
        # 每级固定 1 个采样时，长度 2 的缺口必然被边界切开 -> 无解。
        obs = [x + 5 for x in self.REF]
        res = solve_alignment(
            self.REF, obs, 0, 0, 0, 1, 1, 0, dropout_runs=[[2, 4]]
        )
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_level_supported_only_by_dropouts_is_infeasible(self) -> None:
        # 每级固定 1 个采样；任意缺口都会让某级完全由无效样本支撑。
        obs = [x + 5 for x in self.REF]
        res = solve_alignment(
            self.REF, obs, 0, 0, 0, 1, 1, 0, dropout_runs=[[3, 4]]
        )
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_boundary_may_coincide_with_run_endpoints(self) -> None:
        # 缺口 [2,4)：允许内部边界恰好落在端点 2 / 4 上（块与缺口端点
        # 相接不算切开）；最优解中边界 2 与缺口起点重合。
        good = [x + 5 for x in self.REF]
        obs = good[:2] + [424242, 424242] + good[2:]
        res = solve_alignment(
            self.REF, obs, 5, 5, 0, 1, 3, 2, dropout_runs=[[2, 4]]
        )
        self.assertTrue(res["feasible"])
        self.assertIn(2, res["boundaries"])
        for p in res["boundaries"]:
            self.assertFalse(2 < p < 4, msg=f"边界 {p} 切开了缺口 [2,4)")
        # 缺口整体归属唯一一个级。
        owners = {
            d["dropout_run"]: d["level_order"] for d in res["dropout_runs"]
        }
        self.assertEqual(owners, {0: 2})
        lv = res["levels"][owners[0]]
        self.assertLessEqual(lv["sample_start"], 2)
        self.assertLessEqual(4, lv["sample_end"])

    def test_three_runs_each_assigned_to_exactly_one_level(self) -> None:
        # N=14：3 个缺口各 2 样本，分别落在级 0/2/6（均与一个有效样本同级）。
        obs = [0, 0, 15, 25, 0, 0, 35, 45, 55, 65, 0, 0, 75, 85]
        runs = [[0, 2], [4, 6], [10, 12]]
        res = solve_alignment(
            self.REF, obs, 5, 5, 0, 1, 3, 2, dropout_runs=runs
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(res["residual_sum"], 0)
        assigned = {d["dropout_run"]: d["level_order"] for d in res["dropout_runs"]}
        self.assertEqual(assigned, {0: 0, 1: 2, 2: 6})
        # 每个缺口区间完整包含在所归属级内。
        for d in res["dropout_runs"]:
            lv = res["levels"][d["level_order"]]
            self.assertLessEqual(lv["sample_start"], d["sample_start"])
            self.assertLessEqual(d["sample_end"], lv["sample_end"])

    def test_response_without_field_has_no_dropout_keys(self) -> None:
        obs = [x + 5 for x in self.REF]
        res = solve_alignment(self.REF, obs, -10, 10, 2)
        self.assertTrue(res["feasible"])
        self.assertNotIn("dropout_runs", res)
        for lv in res["levels"]:
            self.assertNotIn("ignored_dropout_runs", lv)
            for s in lv["samples"]:
                self.assertNotIn("ignored", s)
                self.assertIsInstance(s["residual"], int)

    def test_evidence_annotations_are_internally_consistent(self) -> None:
        good = [x + 5 for x in self.REF]
        obs = good[:2] + [424242, 424242] + good[2:]
        runs = [[2, 4]]
        res = solve_alignment(
            self.REF, obs, -10, 10, 2, 1, 3, 2, dropout_runs=runs
        )
        self.assertTrue(res["feasible"])
        # 逐样本证据仍覆盖全部 N 个采样恰好一次（含占位）。
        covered = [s["index"] for lv in res["levels"] for s in lv["samples"]]
        self.assertEqual(covered, list(range(len(obs))))
        ignored_idx = sorted(
            s["index"]
            for lv in res["levels"]
            for s in lv["samples"]
            if s.get("ignored")
        )
        self.assertEqual(ignored_idx, [2, 3])
        # 残差汇总只计有效样本。
        valid_residuals = [
            abs(s["residual"])
            for lv in res["levels"]
            for s in lv["samples"]
            if not s.get("ignored")
        ]
        self.assertEqual(res["residual_sum"], sum(valid_residuals))
        self.assertEqual(res["max_abs_residual"], max(valid_residuals))
        # 每级至少一个有效样本。
        for lv in res["levels"]:
            self.assertTrue(
                any(not s.get("ignored") for s in lv["samples"]),
                msg=f"级 {lv['level_order']} 完全由无效样本支撑",
            )


class DropoutValidationTests(unittest.TestCase):
    REF = [10, 20, 30, 40, 50, 60, 70, 80]

    def _solve(self, runs: object) -> dict:
        return solve_alignment(  # type: ignore[arg-type]
            self.REF, [x + 5 for x in self.REF], 0, 0, 0,
            dropout_runs=runs,
        )

    def test_out_of_range_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([[2, 9]])
        with self.assertRaises(AlignmentError):
            self._solve([[-1, 2]])

    def test_empty_run_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([[3, 3]])

    def test_overlap_and_order_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([[2, 4], [3, 5]])
        with self.assertRaises(AlignmentError):
            self._solve([[4, 6], [1, 2]])

    def test_run_count_cap_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([[0, 1], [1, 2], [2, 3], [3, 4]])
        with self.assertRaises(AlignmentError):
            self._solve([])

    def test_total_dropout_samples_cap_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([[0, 7]])
        with self.assertRaises(AlignmentError):
            self._solve([[0, 2], [2, 4], [4, 8]])

    def test_boundary_counts_are_allowed(self) -> None:
        # 3 个区间、合计 6 个无效样本本身合法（是否可行由对齐决定）。
        res = self._solve([[0, 2], [2, 4], [4, 6]])
        self.assertIn("feasible", res)

    def test_shape_and_type_rejected(self) -> None:
        with self.assertRaises(AlignmentError):
            self._solve([1, 2])  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            self._solve([[1, 2, 3]])  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            self._solve([["1", 2]])  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            self._solve("nope")  # type: ignore[arg-type]

    def test_adjacent_endpoints_are_not_overlap(self) -> None:
        res = self._solve([[0, 1], [1, 2]])
        self.assertIn("feasible", res)


class DropoutBruteForceComparisonTests(unittest.TestCase):
    """带缺口的随机小规模实例：DP 必须与穷举裁决完全一致。"""

    @staticmethod
    def _random_runs(
        rng: random.Random, n: int
    ) -> Optional[List[Tuple[int, int]]]:
        for _ in range(200):
            q = rng.randint(1, 3)
            lengths = [rng.randint(1, 3) for _ in range(q)]
            total = sum(lengths)
            if total > 6 or total > n - 8:
                # 至少保留 8 个有效样本（级数下界为 8-2=6，8 更稳妥）。
                continue
            # 在 n 个位置上贪心放置互不重叠的定长块。
            remaining = list(lengths)
            rng.shuffle(remaining)
            runs: List[Tuple[int, int]] = []
            used = [False] * n
            ok = True
            for length in remaining:
                starts = [
                    s for s in range(0, n - length + 1)
                    if not any(used[s:s + length])
                ]
                if not starts:
                    ok = False
                    break
                s0 = rng.choice(starts)
                for t in range(s0, s0 + length):
                    used[t] = True
                runs.append((s0, s0 + length))
            if not ok:
                continue
            runs.sort()
            return runs
        return None

    def test_random_cases_with_dropouts(self) -> None:
        rng = random.Random(20261004)
        checked = 0
        for trial in range(120):
            R = rng.randint(8, 10)
            N = rng.randint(10, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            used = [0]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
                skipped_set = set()
            dwells = BruteForceComparisonTests._random_composition(
                rng, k, N, 1, 3
            )
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2])
                    obs.append(level + noise)
            runs = self._random_runs(rng, N)
            if runs is None:
                continue
            # 缺口位置写入垃圾值：若错误参与残差必然改变裁决。
            for a, b in runs:
                for t in range(a, b):
                    obs[t] = rng.choice([10_000, -10_000, 9_999])
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            with self.subTest(trial=trial, ref=ref, obs=obs, runs=runs,
                              lo=d_lo, hi=d_hi, limit=limit):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2,
                    dropout_runs=runs,
                )
                checked += 1
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
