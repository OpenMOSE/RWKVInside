"""
Hybrid Attention Selector — DP + Local Search (N-Consecutive Allowed)
====================================================================

- CSV(Alignmentログ)から層ごとの難易度スコアを多角的に算出
- 連続は最大 N 層まで許可（MAX_CONSEC）
- min_gap / max_gap / 理想間隔ペナルティも併用
- 初期解: 動的計画法(DP) + Greedy + Multi-start
- 改善: 焼きなまし(局所探索)でペナルティ付き目的を最大化

Author: GPT-5 (for MASAHIRO SHIMOSE)
"""

import os
import re
import math
import time
import random
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ===================== ユーザ設定 =====================
CSV_PATH    = "stage1log.csv"
NUM_HYBRID  = 4
MIN_GAP     = 1      # 連続を許したいなら 1 を推奨（= 隣接OK）。2以上にするとN連続と矛盾し得ます
MAX_GAP     = 8      # 離れすぎ禁止の目安
MAX_CONSEC  = 2      # ★ ここが新パラメータ：最大で N 層まで連続可（例：2なら 2連続までOK, 3連続はNG）
HEAD_BIAS   = 1.00   # Head側ブースト（1.0〜1.35）
GAP_PENALTY = 0.1   # 理想間隔からの逸脱ペナルティ強度
SA_ITER     = 6000   # 焼きなまし試行回数
SA_T0       = 0.5    # 初期温度
SA_T1       = 0.02   # 終了温度
MULTISTART  = 4      # 初期解の数（DP, Greedy, Random..）
SAVE_DIR    = "./hybrid_pick_results"
os.makedirs(SAVE_DIR, exist_ok=True)
# =====================================================


# ------------------- ユーティリティ -------------------
def _pick_layer_indices(df, suffix):
    """CSVカラムから 'layer_{i}{suffix}' を拾って i 昇順で返す"""
    pat = re.compile(rf"^layer_(\d+){re.escape(suffix)}$")
    pairs = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    return [c for _, c in pairs]

def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _convergence_speed(steps, series):
    """log(rawMSE) ~ a + b*step の b から減衰速度を評価（大ほど速い）"""
    y = np.log(np.clip(series, 1e-8, None))
    if np.allclose(y, y[0]):
        slope = 0.0
    else:
        slope, _, _, _, _ = linregress(steps, y)
    return -slope  # slopeは負が多い → -slope 大 = 速い

def compute_ideal_spacing(L, k):
    return max(1, int(math.floor(L / (k + 1))))

def run_lengths(sol_sorted):
    """昇順の解に対して、連続ブロック長の配列を返す（例： [1,2,1,...]）"""
    if len(sol_sorted) == 0:
        return []
    runs = []
    cur_len = 1
    for a, b in zip(sol_sorted[:-1], sol_sorted[1:]):
        if b == a + 1:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_len = 1
    runs.append(cur_len)
    return runs

def respects_consecutive_limit(sol_sorted, max_consec):
    """最大連続長が max_consec を超えないか"""
    if max_consec is None or max_consec <= 0:
        return True
    return max(run_lengths(sol_sorted), default=0) <= max_consec


# ------------------- データ読み込み＆スコア -------------------
def load_and_score(csv_path, head_bias=HEAD_BIAS):
    df = pd.read_csv(csv_path)
    steps = df["step"].values

    raw_cols   = _pick_layer_indices(df, "_rawMSE")
    scale_cols = _pick_layer_indices(df, "_scale")
    scaled_cols= _pick_layer_indices(df, "_scaledMSE")
    w_cols     = _pick_layer_indices(df, "_balanceW")

    if len(raw_cols) == 0 or len(scale_cols) == 0:
        raise ValueError("CSVに rawMSE または scale の列が見つかりません。")

    # 最小層数に揃える（安全）
    L = min(len(raw_cols), len(scale_cols))
    raw_cols   = raw_cols[:L]
    scale_cols = scale_cols[:L]

    raw_mse   = df[raw_cols].values      # (T, L)
    scale_val = df[scale_cols].values    # (T, L)

    init_raw  = raw_mse[0]               # (L,)
    final_raw = raw_mse[-1]              # (L,)
    align_speed = (init_raw - final_raw) / (init_raw + 1e-8)  # 高いほど速く整合

    conv_speed = np.array([_convergence_speed(steps, raw_mse[:, i]) for i in range(L)])  # 大ほど速い
    final_scale = scale_val[-1]  # (L,)

    # Instability: 時系列微分の分散
    instab = np.array([np.var(np.diff(raw_mse[:, i])) for i in range(L)])

    # Depth bias: Head側優遇（乗算ブースト）
    depth_pos = np.linspace(0.0, 1.0, L)
    head_boost = 1.0 + (head_bias - 1.0) * depth_pos

    # 各指標を「大ほど難しい/優先」に揃える 0..1
    align_slow = 1.0 - _normalize(align_speed)   # 低速ほど 1
    residue    = _normalize(final_raw)           # 残差大ほど 1
    conv_diff  = 1.0 - _normalize(conv_speed)    # 低速（難）ほど 1
    scale_mis  = _normalize(final_scale)         # 大ほど 1
    instab_n   = _normalize(instab)              # 大ほど 1

    # 総合スコア（重みは経験ベース）
    comp = {
        "residue":       0.30 * residue,
        "conv_diff":     0.25 * conv_diff,
        "align_slow":    0.20 * align_slow,
        "scale_mismatch":0.15 * scale_mis,
        "instability":   0.10 * instab_n,
    }
    base = sum(comp.values())
    base = base / (base.max() + 1e-8)
    score = (base * head_boost)
    score = score / (score.max() + 1e-8)

    # 明細テーブル（解析用）
    table = pd.DataFrame({
        "layer": np.arange(L),
        "score": score,
        "residue": residue,
        "conv_diff": conv_diff,
        "align_slow": align_slow,
        "scale_mismatch": scale_mis,
        "instability": instab_n,
        "head_boost": head_boost,
        "final_rawMSE": final_raw,
        "final_scale": final_scale,
        "align_speed_raw": align_speed,
        "conv_speed_raw": conv_speed,
    })
    return L, score, table


# ------------------- ペナルティ・目的関数 -------------------
def spacing_penalty(sol, L, k, min_gap, max_gap, gap_penalty, max_consec):
    """
    解 sol（昇順）に対するペナルティ合計：
      - 近すぎ (<min_gap) は致命的
      - 連続長 > max_consec は致命的
      - 理想間隔からのズレに比例
      - 離れすぎ (>max_gap) は追加ペナルティ
    """
    if len(sol) <= 1:
        return 0.0
    ideal = compute_ideal_spacing(L, k)
    gaps = np.diff(sol)
    pen = 0.0

    # 近すぎ
    if min_gap > 1:
        pen += np.sum((gaps < min_gap) * 1e3)

    # 連続長チェック（max_consec を超えたら致命的）
    if not respects_consecutive_limit(sol, max_consec):
        # 超過ぶんを強く罰する
        runs = run_lengths(sol)
        over = max(0, max(runs) - max_consec)
        pen += 1e3 * over

    # 理想からのズレ
    pen += gap_penalty * np.sum(np.abs(gaps - ideal) / (ideal + 1e-8))

    # 離れすぎ
    pen += np.sum((gaps > max_gap) * 0.75)
    return float(pen)

def objective(sol, score, L, k, min_gap, max_gap, gap_penalty, max_consec):
    """最大化したい目的 = Σ score − ペナルティ"""
    if len(sol) != k:
        return -1e9
    val = float(np.sum(score[sol]))
    pen = spacing_penalty(sol, L, k, min_gap, max_gap, gap_penalty, max_consec)
    return val - pen


# ------------------- Greedy 初期解 -------------------
def greedy_init(score, L, k, min_gap, max_gap, gap_penalty, max_consec):
    ideal = compute_ideal_spacing(L, k)
    cand = list(np.argsort(-score))
    sol = []
    for idx in cand:
        if len(sol) >= k:
            break
        tmp = sorted(sol + [idx])
        # hard constraints
        if (min_gap > 1) and any(abs(idx - j) < min_gap for j in sol):
            continue
        if not respects_consecutive_limit(tmp, max_consec):
            continue
        # soft penalty check
        pen = spacing_penalty(tmp, L, k, min_gap, max_gap, gap_penalty, max_consec)
        if pen > 2.0 * gap_penalty * max(1, k-1):
            continue
        sol = tmp

    # 足りなければ埋める（制約尊重）
    i = 0
    while len(sol) < k and i < L:
        tmp = sorted(sol + [i])
        if ((min_gap == 1) or all(abs(i - j) >= min_gap for j in sol)) and respects_consecutive_limit(tmp, max_consec):
            sol = tmp
        i += 1
    return np.array(sol[:k], dtype=int)


# ------------------- DP（N連続許容版） -------------------
def dp_best(score, L, k, min_gap, max_gap, gap_penalty, max_consec):
    """
    DPで「N連続許容＋間隔ペナルティ」を扱う。
    状態: f(idx, t, last, run_len)
      idx     : 次に見る層インデックス
      t       : 残り選抜数
      last    : 直前に選んだ層（なければ -1）
      run_len : 直前の連続長（last を含む連続ブロック長）
    遷移:
      - skip: f(idx+1, t, last, run_len)
      - take: if gap>=min_gap and new_run_len<=max_consec
              価値: score[idx] - 局所ペナルティ + f(idx+1, t-1, idx, new_run_len)
    """
    from functools import lru_cache
    ideal = compute_ideal_spacing(L, k)

    @lru_cache(maxsize=None)
    def f(idx, t, last, run_len):
        if t == 0:
            return 0.0, ()
        if idx >= L:
            return -1e9, ()

        # skip
        v0, s0 = f(idx + 1, t, last, run_len)

        # take: 連続長と min_gap 判定
        v1, s1 = -1e9, ()
        can_take = True
        local_pen = 0.0

        if last >= 0:
            gap = idx - last
            # min_gap
            if gap < max(1, min_gap):
                can_take = False
            # run_len更新
            new_run = run_len + 1 if gap == 1 else 1
            if new_run > max_consec:
                can_take = False
            # 局所ペナルティ（理想間隔/離れすぎ）
            local_pen += GAP_PENALTY * abs(gap - ideal) / (ideal + 1e-8)
            if gap > max_gap:
                local_pen += 0.75
        else:
            # 最初の1個
            new_run = 1

        if can_take:
            vv, ss = f(idx + 1, t - 1, idx, new_run)
            v1 = score[idx] - local_pen + vv
            s1 = (idx,) + ss

        if v1 > v0:
            return v1, s1
        else:
            return v0, s0

    best_val, best_tuple = f(0, k, -1, 0)
    sol = np.array(best_tuple, dtype=int)
    # DP解が不足することは稀だが安全に埋める
    if len(sol) < k:
        extra = greedy_init(score, L, k, min_gap, max_gap, gap_penalty, max_consec)
        sol = np.unique(np.concatenate([sol, extra]))[:k]
        sol.sort()
    return sol


# ------------------- 多スタート初期解 -------------------
def multi_start_initial_solutions(score, L, k, min_gap, max_gap, gap_penalty, max_consec, n_starts=MULTISTART):
    sols = []
    sols.append(dp_best(score, L, k, min_gap, max_gap, gap_penalty, max_consec))
    sols.append(greedy_init(score, L, k, min_gap, max_gap, gap_penalty, max_consec))

    # ランダム初期解
    for _ in range(max(0, n_starts - len(sols))):
        chosen = []
        attempts = 0
        while len(chosen) < k and attempts < 5 * L:
            idx = random.randrange(L)
            tmp = sorted(chosen + [idx])
            if ((min_gap == 1) or all(abs(idx - j) >= min_gap for j in chosen)) and respects_consecutive_limit(tmp, max_consec):
                chosen = tmp
            attempts += 1
        if len(chosen) < k:
            chosen = greedy_init(score, L, k, min_gap, max_gap, gap_penalty, max_consec)
        sols.append(np.array(chosen[:k], dtype=int))

    # unique
    uniq = []
    seen = set()
    for s in sols:
        key = tuple(s.tolist())
        if key not in seen:
            uniq.append(s)
            seen.add(key)
    return uniq


# ------------------- 近傍操作（連続長制約つき） -------------------
def respects_hard_constraints(sol, min_gap, max_consec):
    if (min_gap > 1) and len(sol) > 1:
        gaps = np.diff(sol)
        if np.any(gaps < min_gap):
            return False
    return respects_consecutive_limit(sol, max_consec)

def neighbors(sol, L, min_gap, max_consec):
    """
    解 sol の近傍を生成（制約尊重）：
      - move: 単一要素を±1〜3へ移動
      - replace: 未使用インデックスに入れ替え（粗いサンプリング）
    """
    sol = np.array(sol, dtype=int)
    used = set(sol.tolist())
    cand = []

    # move
    for i in range(len(sol)):
        for delta in (-3, -2, -1, 1, 2, 3):
            j = sol[i] + delta
            if 0 <= j < L and j not in used:
                tmp = sol.copy()
                tmp[i] = j
                tmp.sort()
                if respects_hard_constraints(tmp, min_gap, max_consec):
                    cand.append(tmp)

    # replace
    stride = max(1, L // 32)
    for i in range(len(sol)):
        for j in range(0, L, stride):
            if j in used:
                continue
            tmp = sol.copy()
            tmp[i] = j
            tmp.sort()
            if respects_hard_constraints(tmp, min_gap, max_consec):
                cand.append(tmp)

    # dedup
    uniq, seen = [], set()
    for s in cand:
        key = tuple(s.tolist())
        if key not in seen:
            uniq.append(s)
            seen.add(key)
    return uniq


# ------------------- 焼きなまし（局所探索） -------------------
def anneal(score, L, k, min_gap, max_gap, gap_penalty, max_consec,
           init_sol, iters=SA_ITER, T0=SA_T0, T1=SA_T1):
    rng = random.Random(42 + int(time.time()) % 10000)
    best = init_sol.copy()
    best_val = objective(best, score, L, k, min_gap, max_gap, gap_penalty, max_consec)

    cur = best.copy()
    cur_val = best_val

    for it in range(1, iters + 1):
        T = T0 * (T1 / T0) ** (it / iters)  # 幾何冷却
        nb_list = neighbors(cur, L, min_gap, max_consec)
        if not nb_list:
            break
        cand = rng.choice(nb_list)
        val = objective(cand, score, L, k, min_gap, max_gap, gap_penalty, max_consec)
        if val > cur_val:
            cur, cur_val = cand, val
            if val > best_val:
                best, best_val = cand, val
        else:
            if rng.random() < math.exp((val - cur_val) / max(1e-8, T)):
                cur, cur_val = cand, val

    return best, best_val


# ------------------- スコア作成と最適化 -------------------
def load_and_optimize():
    L, score, table = load_and_score(CSV_PATH, head_bias=HEAD_BIAS)
    ideal = compute_ideal_spacing(L, NUM_HYBRID)

    # 初期解（複数）
    inits = multi_start_initial_solutions(
        score, L, NUM_HYBRID, MIN_GAP, MAX_GAP, GAP_PENALTY, MAX_CONSEC, n_starts=MULTISTART
    )

    # それぞれを焼きなましで磨く
    best_sol, best_val = None, -1e9
    for s in inits:
        loc_best, loc_val = anneal(
            score, L, NUM_HYBRID, MIN_GAP, MAX_GAP, GAP_PENALTY, MAX_CONSEC,
            init_sol=s, iters=SA_ITER, T0=SA_T0, T1=SA_T1
        )
        if loc_val > best_val:
            best_sol, best_val = loc_best, loc_val

    best_sol = np.array(sorted(best_sol), dtype=int)

    # レポート
    gaps = np.diff(best_sol) if len(best_sol) > 1 else np.array([])
    report = []
    report.append("=== Hybrid Attention Selection — DP + Annealing (N-Consecutive Allowed) ===")
    report.append(f"CSV: {CSV_PATH}")
    report.append(f"Layers (L): {L}")
    report.append(f"Target Hybrid: {NUM_HYBRID}")
    report.append(f"MAX_CONSEC: {MAX_CONSEC}")
    report.append(f"Selected: {best_sol.tolist()}")
    if gaps.size:
        report.append(f"Gaps: avg={gaps.mean():.2f}, min={gaps.min()}, max={gaps.max()}, ideal≈{ideal}")
    report.append(f"Objective (score - penalty): {best_val:.4f}")
    report.append("")
    report.append("-- Top 10 layers by base score --")
    top_idx = np.argsort(-score)[:min(10, L)]
    for r, i in enumerate(top_idx, 1):
        report.append(f"{r:>3}: layer {i:>2} | base_score={score[i]:.3f}")
    txt = "\n".join(report)
    print(txt)
    with open(os.path.join(SAVE_DIR, "selection_report.txt"), "w") as f:
        f.write(txt + "\n")

    # スコア明細を保存（解析容易化）
    out = table.copy()
    out["picked"] = 0
    out.loc[out["layer"].isin(best_sol.tolist()), "picked"] = 1
    out.to_csv(os.path.join(SAVE_DIR, "layer_scores.csv"), index=False)

    print("\nSaved:", os.path.join(SAVE_DIR, "selection_report.txt"))
    print("Saved:", os.path.join(SAVE_DIR, "layer_scores.csv"))
    print("Suggested attention layers:", best_sol.tolist())


if __name__ == "__main__":
    load_and_optimize()
