# -*- coding: utf-8 -*-
"""
demo_core.py — phần "não" của live demo Naive Bayes vs Decision Tree.

Tách riêng khỏi giao diện (app.py) để:
  * dễ đọc: sinh viên mở file này là thấy Naive Bayes thực chất chỉ là
    "đếm tần suất + nhân xác suất" (NaiveBayesDemo, ~60 dòng, không dùng hộp đen)
  * dễ kiểm thử: có thể import và chạy không cần Streamlit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

RANDOM_STATE = 42

# ---------------------------------------------------------------- features
CAT_COLS = ["job", "marital", "education", "contact", "month", "poutcome"]
BIN_COLS = ["default", "housing", "loan"]            # yes / no
NUM_COLS = ["age", "balance", "duration", "campaign", "previous"]

LABEL = {
    "age": "Tuổi", "job": "Nghề nghiệp", "marital": "Hôn nhân", "education": "Học vấn",
    "default": "Có nợ xấu", "balance": "Số dư tài khoản (€)", "housing": "Vay mua nhà",
    "loan": "Vay cá nhân", "contact": "Kênh liên hệ", "month": "Tháng liên hệ",
    "duration": "Thời lượng cuộc gọi (giây)", "campaign": "Số lần gọi (đợt này)",
    "previous": "Số lần liên hệ trước đó", "poutcome": "Kết quả chiến dịch trước",
}
YESNO_VI = {"yes": "Có", "no": "Không"}


def show_value(feat, v, kind="cat") -> str:
    """Giá trị hiển thị: yes/no -> Có/Không; số -> số nguyên."""
    if kind == "num":
        return f"{int(round(float(v)))}"
    return YESNO_VI.get(v, v) if feat in BIN_COLS else str(v)

# màu thống nhất toàn demo
C_YES, C_NO = "#2A9D8F", "#E76F51"      # đăng ký / không đăng ký
C_NB, C_DT = "#DD8452", "#4C72B0"       # hai mô hình


# ---------------------------------------------------------------- định dạng kiểu Việt Nam
def vn_num(x, nd=0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def vn_pct(p, nd=1) -> str:
    return vn_num(p * 100, nd) + "%"


# ---------------------------------------------------------------- dữ liệu
def load_bank(source) -> pd.DataFrame:
    """Đọc bank-full.csv (chấp nhận cả dấu ',' lẫn ';' của bản UCI gốc)."""
    df = pd.read_csv(source)
    if df.shape[1] == 1:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def feature_lists(use_duration: bool):
    cats = CAT_COLS + BIN_COLS
    nums = [c for c in NUM_COLS if use_duration or c != "duration"]
    return cats, nums


def split_data(df: pd.DataFrame):
    """Chia 80/20, stratify theo y — cùng seed với pipeline của TV2."""
    idx_tr, idx_te = train_test_split(
        df.index, test_size=0.2, stratify=df["y"], random_state=RANDOM_STATE)
    return df.loc[idx_tr], df.loc[idx_te]


# ================================================================
#  NAIVE BAYES viết tay — không hộp đen
# ================================================================
class NaiveBayesDemo:
    """
    Naive Bayes hỗn hợp:
      * feature phân loại : P(giá trị | lớp) = (đếm + 1) / (tổng + K)   (làm mịn Laplace)
      * feature số        : P(giá trị | lớp) ~ phân phối chuẩn N(mean_lớp, std_lớp)
    Dự đoán:  odds(yes) = odds_nền × Π (hệ số bằng chứng từng feature)
    """

    def __init__(self, cat_cols, num_cols, alpha: float = 1.0):
        self.cat_cols, self.num_cols, self.alpha = list(cat_cols), list(num_cols), alpha

    # ---- học: chỉ là đếm và tính trung bình
    def fit(self, df: pd.DataFrame, y):
        y = np.asarray(y)
        self.n_ = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
        self.prior_ = {c: self.n_[c] / len(y) for c in (0, 1)}

        self.cat_counts_, self.cat_prob_ = {}, {}
        for col in self.cat_cols:
            counts = pd.crosstab(df[col], y).reindex(columns=[0, 1], fill_value=0)
            k = len(counts)
            self.cat_counts_[col] = counts
            self.cat_prob_[col] = (counts + self.alpha) / (counts.sum(axis=0) + self.alpha * k)

        self.num_stats_ = {}
        for col in self.num_cols:
            g = df[col].astype(float).groupby(y)
            mu, sd = g.mean(), g.std(ddof=0).clip(lower=1e-6)
            self.num_stats_[col] = {c: (float(mu[c]), float(sd[c])) for c in (0, 1)}
        return self

    @staticmethod
    def _logpdf(x, mu, sd):
        return -0.5 * np.log(2 * np.pi * sd ** 2) - (x - mu) ** 2 / (2 * sd ** 2)

    @property
    def prior_log_odds(self) -> float:
        return float(np.log(self.prior_[1] / self.prior_[0]))

    # ---- dự đoán hàng loạt (vector hoá)
    def log_ratio_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.cat_cols:
            lp = np.log(self.cat_prob_[col])
            out[col] = df[col].map(lp[1] - lp[0]).fillna(0.0).to_numpy()
        for col in self.num_cols:
            (m1, s1), (m0, s0) = self.num_stats_[col][1], self.num_stats_[col][0]
            x = df[col].to_numpy(float)
            out[col] = self._logpdf(x, m1, s1) - self._logpdf(x, m0, s0)
        return pd.DataFrame(out, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        lo = self.prior_log_odds + self.log_ratio_frame(df).sum(axis=1).to_numpy()
        return expit(lo)

    # ---- giải thích 1 khách hàng: bảng từng feature "bỏ phiếu" thế nào
    def explain(self, row) -> pd.DataFrame:
        rows = []
        for col in self.cat_cols:
            v = row[col]
            pr, cn = self.cat_prob_[col], self.cat_counts_[col]
            p1, p0 = float(pr.loc[v, 1]), float(pr.loc[v, 0])
            rows.append(dict(feature=col, value=v, kind="cat", p_yes=p1, p_no=p0,
                             log_ratio=float(np.log(p1 / p0)),
                             count_yes=int(cn.loc[v, 1]), count_no=int(cn.loc[v, 0])))
        for col in self.num_cols:
            x = float(row[col])
            (m1, s1), (m0, s0) = self.num_stats_[col][1], self.num_stats_[col][0]
            l1, l0 = self._logpdf(x, m1, s1), self._logpdf(x, m0, s0)
            rows.append(dict(feature=col, value=x, kind="num", p_yes=float(np.exp(l1)),
                             p_no=float(np.exp(l0)), log_ratio=float(l1 - l0),
                             mean_yes=m1, sd_yes=s1, mean_no=m0, sd_no=s0))
        e = pd.DataFrame(rows)
        order = e["log_ratio"].abs().sort_values(ascending=False).index
        return e.loc[order].reset_index(drop=True)

    def evolution(self, expl: pd.DataFrame) -> np.ndarray:
        """P(yes) sau mỗi bước cộng bằng chứng; phần tử 0 = xác suất nền."""
        lo = self.prior_log_odds + np.concatenate([[0.0], expl["log_ratio"].cumsum().to_numpy()])
        return expit(lo)


# ================================================================
#  DECISION TREE — mã hoá + diễn giải đường đi
# ================================================================
def encode_tree(df, cat_cols, bin_cols, num_cols, columns=None) -> pd.DataFrame:
    """One-hot cho cột phân loại (tên dạng 'job=management'), yes/no -> 1/0."""
    parts = [df[num_cols].astype(float),
             pd.DataFrame({c: (df[c] == "yes").astype(int) for c in bin_cols}, index=df.index),
             pd.get_dummies(df[cat_cols], prefix_sep="=").astype(int)]
    X = pd.concat(parts, axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def _is_numeric_col(col: str) -> bool:
    return "=" not in col and col not in BIN_COLS


def pretty_feature(col: str) -> str:
    if "=" in col:
        base, cat = col.split("=", 1)
        return f"{LABEL[base]} = {cat}"
    if col in BIN_COLS:
        return f"{LABEL[col]} = có"
    return LABEL[col]


def tree_question(col: str, thresh: float) -> str:
    if "=" in col:
        base, cat = col.split("=", 1)
        return f'{LABEL[base]} là "{cat}"?'
    if col in BIN_COLS:
        return f"{LABEL[col]}?"
    return f"{LABEL[col]} ≤ {int(np.floor(thresh))}?"


def _leaf_stats(tree, node):
    v = tree.value[node][0]
    return int(tree.n_node_samples[node]), float(v[1] / v.sum())


def train_tree(X, y, depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(X, y)


def tree_path(dt, columns, x_row: pd.DataFrame):
    """Đi từ gốc tới lá, ghi lại từng câu hỏi + câu trả lời của khách hàng."""
    t, node, steps = dt.tree_, 0, []
    while t.children_left[node] != -1:
        col, th = columns[t.feature[node]], t.threshold[node]
        val = float(x_row.iloc[0][col])
        go_left = val <= th
        answer_yes = go_left if _is_numeric_col(col) else (not go_left)
        steps.append(dict(node=node, question=tree_question(col, th), answer=bool(answer_yes),
                          value=(f"{val:.0f}" if _is_numeric_col(col) else "")))
        node = t.children_left[node] if go_left else t.children_right[node]
    n, p = _leaf_stats(t, node)
    return steps, node, n, p


def _node_color(p: float) -> str:
    lo, mid, hi = np.array([232, 146, 83]), np.array([255, 255, 255]), np.array([70, 140, 210])
    c = mid + (lo - mid) * (0.5 - p) / 0.5 if p < 0.5 else mid + (hi - mid) * (p - 0.5) / 0.5
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def tree_to_dot(dt, columns, path_nodes=()) -> str:
    """DOT tự viết: nhãn tiếng Việt, cạnh 'Có/Không', tô đỏ đường đi của khách hàng."""
    t, path = dt.tree_, set(path_nodes)
    esc = lambda s: s.replace('"', '\\"')
    L = ["digraph T {", "rankdir=TB;",
         'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, margin="0.12,0.08"];',
         'edge [fontname="Helvetica", fontsize=12];']
    for i in range(t.node_count):
        n, p = _leaf_stats(t, i)
        if t.children_left[i] == -1:
            label = f"P(đăng ký) = {p*100:.0f}%\\n{vn_num(n)} khách"
        else:
            label = f"{esc(tree_question(columns[t.feature[i]], t.threshold[i]))}\\n({vn_num(n)} khách · {p*100:.0f}% đăng ký)"
        attrs = f'label="{label}", fillcolor="{_node_color(p)}"'
        if i in path:
            attrs += ', color="#C0392B", penwidth=4'
        L.append(f"{i} [{attrs}];")
    for i in range(t.node_count):
        l, r = t.children_left[i], t.children_right[i]
        if l == -1:
            continue
        numeric = _is_numeric_col(columns[t.feature[i]])
        yes_child, no_child = (l, r) if numeric else (r, l)
        for child, text in ((yes_child, "Có"), (no_child, "Không")):
            extra = ', color="#C0392B", penwidth=4, fontcolor="#C0392B"' if (i in path and child in path) else ""
            L.append(f'{i} -> {child} [label="{text}"{extra}];')
    L.append("}")
    return "\n".join(L)


def depth_curve(Xtr, ytr, Xte, yte, depths=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)) -> pd.DataFrame:
    rows = []
    for d in depths:
        m = train_tree(Xtr, ytr, d)
        rows.append(dict(depth=d, train=m.score(Xtr, ytr), test=m.score(Xte, yte)))
    return pd.DataFrame(rows)


# ================================================================
#  ĐÁNH GIÁ & ỨNG DỤNG (xếp hạng khách hàng)
# ================================================================
def metrics_at_threshold(y_true, score, thr=0.5) -> dict:
    pred = (score >= thr).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                auc=roc_auc_score(y_true, score))


def gains_curve(y_true, score):
    order = np.argsort(-score, kind="stable")
    ys = np.asarray(y_true)[order]
    x = np.arange(1, len(ys) + 1) / len(ys) * 100
    g = np.cumsum(ys) / ys.sum() * 100
    return np.concatenate([[0], x]), np.concatenate([[0], g])


def captured_at(x, g, pct) -> float:
    return float(np.interp(pct, x, g))


# ================================================================
#  BIỂU ĐỒ (matplotlib) — cỡ chữ lớn để chiếu lớp học
# ================================================================
def _style():
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})


def fig_nb_votes(expl: pd.DataFrame, k: int):
    """Mỗi feature = 1 lá phiếu với 'hệ số ×'. Xét tới feature thứ k; phần còn lại mờ đi."""
    _style()
    n = len(expl)
    fig, ax = plt.subplots(figsize=(9, 0.42 * n + 1.4))
    mult = np.exp(expl["log_ratio"].to_numpy())
    for i in range(n):
        active = i < k
        col = C_YES if mult[i] >= 1 else C_NO
        ax.barh(i, np.log(mult[i]), color=col, alpha=1.0 if active else 0.13, height=0.62)
        if active:
            txt = f"×{mult[i]:.2f}" if 0.005 < mult[i] < 1000 else ("×>1000" if mult[i] >= 1000 else "×<0,005")
            ax.text(np.log(mult[i]) + (0.06 if mult[i] >= 1 else -0.06), i, txt.replace(".", ","),
                    va="center", ha="left" if mult[i] >= 1 else "right", fontsize=11)
    labels = [f"{LABEL[f]} = {show_value(f, v, kind)}" for f, v, kind in
              zip(expl["feature"], expl["value"], expl["kind"])]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    lim = max(1.5, float(np.abs(np.log(mult)).max()) * 1.35)
    ax.set_xlim(-lim, lim)
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    ax.set_xticks([np.log(t) for t in ticks if abs(np.log(t)) < lim])
    ax.set_xticklabels([f"×{t}".replace(".", ",") for t in ticks if abs(np.log(t)) < lim])
    ax.axvline(0, color="#555555", linewidth=1.3)
    ax.set_xlabel("← nghiêng KHÔNG đăng ký   |   nghiêng ĐĂNG KÝ →")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_feature_detail(nb: NaiveBayesDemo, expl_row, train_df: pd.DataFrame, y_train):
    """Xem 'P(giá trị | lớp)' của 1 feature: cột (phân loại) hoặc đường cong chuẩn (số)."""
    _style()
    y_train = np.asarray(y_train)
    feat, val = expl_row["feature"], expl_row["value"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    if expl_row["kind"] == "cat":
        pr = nb.cat_prob_[feat]
        order = pr[1].sort_values(ascending=False).index.tolist()
        xs = np.arange(len(order))
        ax.bar(xs - 0.2, pr.loc[order, 1] * 100, 0.4, color=C_YES, label="Trong nhóm ĐĂNG KÝ")
        ax.bar(xs + 0.2, pr.loc[order, 0] * 100, 0.4, color=C_NO, label="Trong nhóm KHÔNG đăng ký")
        j = order.index(val)
        ax.axvspan(j - 0.5, j + 0.5, color="#FFE9A8", alpha=0.7, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([show_value(feat, o) for o in order], rotation=35, ha="right")
        ax.set_ylabel("% khách hàng có giá trị này")
        ax.set_title(f"{LABEL[feat]}: khách hàng thuộc nhóm “{show_value(feat, val)}” (vùng vàng)")
    else:
        v_yes, v_no = train_df.loc[y_train == 1, feat].to_numpy(float), train_df.loc[y_train == 0, feat].to_numpy(float)
        lo, hi = np.percentile(np.r_[v_yes, v_no], [0.5, 97.5])
        hi = max(hi, val * 1.1)
        grid = np.linspace(min(lo, val), hi, 400)
        ax.hist(v_no, bins=40, range=(grid[0], hi), density=True, color=C_NO, alpha=0.18)
        ax.hist(v_yes, bins=40, range=(grid[0], hi), density=True, color=C_YES, alpha=0.18)
        pdf = lambda x, m, s: np.exp(nb._logpdf(x, m, s))
        ax.plot(grid, pdf(grid, expl_row["mean_yes"], expl_row["sd_yes"]), color=C_YES, lw=2.8, label="NB giả định: nhóm ĐĂNG KÝ")
        ax.plot(grid, pdf(grid, expl_row["mean_no"], expl_row["sd_no"]), color=C_NO, lw=2.8, label="NB giả định: nhóm KHÔNG")
        ax.axvline(val, color="#333333", lw=2, ls="--")
        ax.scatter([val, val], [expl_row["p_yes"], expl_row["p_no"]], color=[C_YES, C_NO], s=90, zorder=5, edgecolor="black")
        ax.set_xlabel(LABEL[feat])
        ax.set_ylabel("mật độ (độ “hợp lý”)")
        ax.set_title(f"{LABEL[feat]} = {int(val)} (nét đứt); hai chấm = likelihood")
        ax.text(0.98, 0.5, "cột mờ = dữ liệu thật\nđường cong = giả định NB", transform=ax.transAxes,
                ha="right", va="center", fontsize=10.5, color="#555555")
    ax.legend(fontsize=10.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_double_counting(nb: NaiveBayesDemo, expl: pd.DataFrame, feat: str, p_tree: float, kmax=10):
    """Thí nghiệm: lặp lại 1 thông tin k lần — NB 'tin' k lần, cây quyết định thì không."""
    _style()
    lr = float(expl.loc[expl["feature"] == feat, "log_ratio"].iloc[0])
    rest = float(expl["log_ratio"].sum() - lr)
    ks = np.arange(1, kmax + 1)
    p_nb = expit(nb.prior_log_odds + rest + ks * lr)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ks, p_nb * 100, "-o", color=C_NB, lw=2.8, label="Naive Bayes")
    ax.axhline(p_tree * 100, color=C_DT, lw=2.8, ls="--", label="Decision Tree (không đổi)")
    ax.axhline(50, color="#999999", lw=1, ls=":")
    ax.annotate(f"{p_nb[0]*100:.1f}%".replace(".", ","), (1, p_nb[0] * 100), textcoords="offset points", xytext=(6, 8), color=C_NB)
    last = "<0,1%" if p_nb[-1] < 0.001 else (f"{p_nb[-1]*100:.1f}%".replace(".", ","))
    ax.annotate(last, (kmax, p_nb[-1] * 100), textcoords="offset points", xytext=(-40, 10), color=C_NB)
    ax.set_xlabel("Số lần ta lặp lại đúng 1 thông tin (bản sao của cùng 1 cột)")
    ax.set_ylabel("P(đăng ký) %")
    ax.set_xticks(ks)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    return fig


def fig_overfit(curve: pd.DataFrame, current_depth: int, baseline: float):
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(curve["depth"], curve["train"] * 100, "--o", color="#8FA9C7", lw=2.4, label="Độ chính xác trên tập HỌC")
    ax.plot(curve["depth"], curve["test"] * 100, "-o", color=C_DT, lw=2.8, label="Độ chính xác trên tập TEST (chưa từng thấy)")
    ax.axhline(baseline * 100, color="#999999", ls=":", lw=1.4)
    ax.text(curve["depth"].max(), baseline * 100 + 0.25, "luôn đoán “KHÔNG”", ha="right", color="#777777", fontsize=10.5)
    ax.axvline(current_depth, color=C_NB, lw=2.2, alpha=0.8)
    ax.text(current_depth + 0.25, ax.get_ylim()[0] + 0.8, "độ sâu đang chọn", color=C_NB, fontsize=10.5)
    ax.set_xlabel("Độ sâu tối đa của cây (max_depth)")
    ax.set_ylabel("Accuracy %")
    ax.set_xticks([1, 3, 5, 8, 10, 12, 15, 20])
    ax.legend(fontsize=10.5, loc="upper left")
    fig.tight_layout()
    return fig


def fig_importance(dt, columns, top=8):
    _style()
    imp = pd.Series(dt.feature_importances_, index=columns).sort_values(ascending=False).head(top)
    imp = imp[imp > 0]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * max(len(imp), 2) + 1.2))
    ax.barh([pretty_feature(c) for c in imp.index], imp.values * 100, color=C_DT)
    ax.invert_yaxis()
    ax.set_xlabel("Mức độ quan trọng (%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_gains(x_nb, g_nb, x_dt, g_dt, budget, pos_nb=None, pos_dt=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.plot(x_dt, g_dt, color=C_DT, lw=2.8, label="Decision Tree")
    ax.plot(x_nb, g_nb, color=C_NB, lw=2.8, label="Naive Bayes")
    ax.plot([0, 100], [0, 100], "--", color="#999999", lw=1.8, label="Gọi ngẫu nhiên")
    ax.axvline(budget, color="#666666", lw=1.2, ls=":")
    v_dt, v_nb = captured_at(x_dt, g_dt, budget), captured_at(x_nb, g_nb, budget)
    for v, other, c in ((v_dt, v_nb, C_DT), (v_nb, v_dt, C_NB)):
        ax.scatter([budget], [v], color=c, s=70, zorder=5)
        dy = 10 if v >= other else -22          # đường cao ghi phía trên, đường thấp ghi phía dưới
        ax.annotate(f"{v:.0f}%", (budget, v), textcoords="offset points", xytext=(10, dy), color=c, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    for pos, c, name in ((pos_nb, C_NB, "NB"), (pos_dt, C_DT, "DT")):
        if pos is not None:
            ax.scatter([pos], [0], marker="v", s=110, color=c, zorder=6, clip_on=False)
    ax.set_xlabel("% khách hàng được gọi (xếp từ điểm cao → thấp)")
    ax.set_ylabel("% tổng số người ĐĂNG KÝ tìm được")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    return fig
