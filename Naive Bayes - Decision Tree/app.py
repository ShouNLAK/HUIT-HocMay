# -*- coding: utf-8 -*-
"""
LIVE DEMO — Naive Bayes vs Decision Tree: "xem máy suy nghĩ như thế nào"
Chạy:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")
import demo_core as core
from demo_core import vn_num, vn_pct

st.set_page_config(page_title="Naive Bayes vs Decision Tree — Live Demo",
                   page_icon="🏦", layout="wide")

HERE = Path(__file__).parent
EXAMPLE_IDX = 12657          # khách hàng ví dụ dùng xuyên suốt (giống slide TV2)

# ------------------------------------------------------------------ dữ liệu
@st.cache_data(show_spinner="Đang đọc dữ liệu…")
def load_default():
    for p in (HERE / "data" / "bank-full.csv", HERE / "bank-full.csv"):
        if p.exists():
            return core.load_bank(p)
    return None


df = load_default()
if df is None:
    st.warning("Không tìm thấy `data/bank-full.csv`. Hãy tải lên file dữ liệu Bank Marketing:")
    up = st.file_uploader("bank-full.csv", type="csv")
    if up is None:
        st.stop()
    df = core.load_bank(up)

N_ROWS = len(df)


# ------------------------------------------------------------------ mô hình (cache)
@st.cache_resource(show_spinner="Đang huấn luyện mô hình…")
def get_bundle(n_rows: int, use_duration: bool, _df):
    tr, te = core.split_data(_df)
    cats, nums = core.feature_lists(use_duration)
    nb = core.NaiveBayesDemo(cats, nums).fit(tr, tr["y"])
    Xtr = core.encode_tree(tr, core.CAT_COLS, core.BIN_COLS, nums)
    cols = list(Xtr.columns)
    Xte = core.encode_tree(te, core.CAT_COLS, core.BIN_COLS, nums, cols)
    return dict(train=tr, test=te, cats=cats, nums=nums, nb=nb, Xtr=Xtr, Xte=Xte,
                cols=cols, p_nb=nb.predict_proba(te))


@st.cache_resource(show_spinner="Đang huấn luyện cây quyết định…")
def get_tree(n_rows: int, use_duration: bool, depth: int, _b):
    dt = core.train_tree(_b["Xtr"], _b["train"]["y"], depth)
    return dt, dt.predict_proba(_b["Xte"])[:, 1]


@st.cache_data(show_spinner="Đang vẽ đường cong overfitting…")
def get_curve(n_rows: int, use_duration: bool, _b):
    return core.depth_curve(_b["Xtr"], _b["train"]["y"], _b["Xte"], _b["test"]["y"])


@st.cache_data(show_spinner=False)
def get_presets(n_rows: int, _df):
    """Chọn khách THẬT từ tập test để làm ví dụ (không bịa số liệu)."""
    b = get_bundle(n_rows, True, _df)
    dt, p_dt = get_tree(n_rows, True, 3, b)
    te = b["test"]
    p_nb = pd.Series(b["p_nb"], index=te.index)
    p_dt = pd.Series(p_dt, index=te.index)
    hi = te[(te["y"] == 1) & (te["poutcome"] == "success") & (te["duration"] > 300)].index[0]
    hard = te[(te["y"] == 0) & te["campaign"].between(6, 9)].index[0]
    # khách "bất đồng": NB gom nhiều bằng chứng nhỏ -> bắt được người đăng ký, còn cây bỏ sót
    cand = te[(p_nb >= 0.5) & (p_nb < 0.9) & (p_dt < 0.3) & (te["y"] == 1)].index
    dis = (p_nb - p_dt)[cand].idxmax() if len(cand) else te.index[1]
    return {
        "① Khách ví dụ: 27 tuổi, quản lý, chưa từng được liên hệ": EXAMPLE_IDX,
        "② Khách tiềm năng cao (đã từng đăng ký ở chiến dịch trước)": int(hi),
        "③ Khách bị gọi rất nhiều lần": int(hard),
        "④ Hai mô hình BẤT ĐỒNG ý kiến": int(dis),
    }


PRESETS = get_presets(N_ROWS, df)

# ------------------------------------------------------------------ widget: khách hàng
CAT_OPTIONS = {c: sorted(df[c].unique().tolist()) for c in core.CAT_COLS}
RANGES = {"age": (18, 95), "balance": (-9000, 110000), "duration": (0, 1500),
          "campaign": (1, 20), "previous": (0, 15)}
ALL_INPUTS = core.CAT_COLS + core.BIN_COLS + core.NUM_COLS


def _clip(col, v):
    lo, hi = RANGES.get(col, (None, None))
    return v if lo is None else int(min(max(int(v), lo), hi))


def apply_preset():
    r = df.loc[PRESETS[st.session_state["preset"]]]
    for c in ALL_INPUTS:
        v = r[c]
        st.session_state[f"in_{c}"] = _clip(c, v) if c in core.NUM_COLS else v


if "preset" not in st.session_state:
    st.session_state["preset"] = list(PRESETS)[0]
    apply_preset()

with st.sidebar:
    st.header("🏦 Khách hàng")
    st.selectbox("Chọn khách mẫu (dữ liệu thật)", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("…hoặc tự chỉnh từng thông tin bên dưới và xem hai mô hình đổi ý ngay lập tức.")

    st.slider(core.LABEL["age"], *RANGES["age"], key="in_age")
    st.selectbox(core.LABEL["job"], CAT_OPTIONS["job"], key="in_job")
    st.selectbox(core.LABEL["marital"], CAT_OPTIONS["marital"], key="in_marital")
    st.selectbox(core.LABEL["education"], CAT_OPTIONS["education"], key="in_education")
    st.number_input(core.LABEL["balance"], RANGES["balance"][0], RANGES["balance"][1], step=100, key="in_balance")
    for c in core.BIN_COLS:
        st.selectbox(core.LABEL[c], ["no", "yes"], key=f"in_{c}", format_func=lambda v: core.YESNO_VI[v])
    st.selectbox(core.LABEL["contact"], CAT_OPTIONS["contact"], key="in_contact")
    st.selectbox(core.LABEL["month"], CAT_OPTIONS["month"], key="in_month")
    st.slider(core.LABEL["duration"], *RANGES["duration"], step=5, key="in_duration",
              disabled=not st.session_state.get("use_duration", True))
    st.slider(core.LABEL["campaign"], *RANGES["campaign"], key="in_campaign")
    st.slider(core.LABEL["previous"], *RANGES["previous"], key="in_previous")
    st.selectbox(core.LABEL["poutcome"], CAT_OPTIONS["poutcome"], key="in_poutcome")

    st.divider()
    st.header("⚙️ Mô hình")
    use_duration = st.checkbox("Dùng “thời lượng cuộc gọi” (duration)", value=True, key="use_duration",
                               help="Thời lượng chỉ biết SAU khi cuộc gọi kết thúc. Bỏ chọn để mô phỏng "
                                    "dự đoán TRƯỚC khi gọi — thực tế hơn nhưng mô hình yếu đi rõ rệt.")
    depth = st.slider("Độ sâu tối đa của cây quyết định", 1, 5, 3, key="depth")

# ------------------------------------------------------------------ tính toán cho khách hàng hiện tại
B = get_bundle(N_ROWS, use_duration, df)
DT, P_DT_TEST = get_tree(N_ROWS, use_duration, depth, B)
nb, cats, nums, cols = B["nb"], B["cats"], B["nums"], B["cols"]
te, tr = B["test"], B["train"]

row = {c: st.session_state[f"in_{c}"] for c in ALL_INPUTS}
x_df = pd.DataFrame([row])
p_nb = float(nb.predict_proba(x_df)[0])
x_tree = core.encode_tree(x_df, core.CAT_COLS, core.BIN_COLS, nums, cols)
p_dt = float(DT.predict_proba(x_tree)[0, 1])
expl = nb.explain(row)
evo = nb.evolution(expl)
steps, leaf, leaf_n, leaf_p = core.tree_path(DT, cols, x_tree)

# đáp án thật (chỉ khi thông tin đang khớp 100% với khách mẫu đã chọn)
truth = None
_r = df.loc[PRESETS[st.session_state["preset"]]]
if all((_clip(c, _r[c]) if c in core.NUM_COLS else _r[c]) == row[c] for c in ALL_INPUTS):
    truth = int(_r["y"])


def show_tree(dot: str):
    try:
        st.graphviz_chart(dot, width="stretch")
    except TypeError:                       # Streamlit cũ
        st.graphviz_chart(dot, use_container_width=True)


def fig_out(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ------------------------------------------------------------------ tiêu đề
st.title("🏦 Naive Bayes vs Decision Tree — xem máy “suy nghĩ” thế nào")
st.caption(f"Dữ liệu thật: {vn_num(N_ROWS)} khách hàng của một ngân hàng (UCI Bank Marketing). "
           f"Câu hỏi: khách hàng này có đăng ký tiền gửi có kỳ hạn sau cuộc gọi tiếp thị không?")

tab_res, tab_nb, tab_dt, tab_biz = st.tabs(
    ["🎯 Kết quả", "🧮 Naive Bayes", "🌳 Decision Tree", "📞 Ứng dụng: gọi ai trước?"])

# ================================================================== TAB 1 — KẾT QUẢ
with tab_res:
    r = row
    st.markdown(
        f"**Khách hàng đang xét:** {r['age']} tuổi · {r['job']} · {r['marital']} · học vấn {r['education']} · "
        f"số dư {vn_num(r['balance'])}€ · vay nhà: {core.YESNO_VI[r['housing']]} · vay cá nhân: {core.YESNO_VI[r['loan']]} · "
        f"liên hệ qua {r['contact']} (tháng {r['month']})"
        + (f" · gọi {r['duration']} giây" if use_duration else "")
        + f" · {r['campaign']} lần gọi trong đợt này · đã liên hệ trước đó {r['previous']} lần (kết quả: {r['poutcome']})")

    c1, c2 = st.columns(2)
    for col, name, p, base in ((c1, "🧮 Naive Bayes", p_nb, nb.prior_[1]),
                               (c2, "🌳 Decision Tree", p_dt, float(tr["y"].mean()))):
        with col:
            st.subheader(name)
            st.metric("Xác suất đăng ký", vn_pct(p), delta=f"{(p - base) * 100:+.1f} điểm % so với mức nền".replace(".", ","))
            st.progress(float(min(max(p, 0.0), 1.0)))
            (st.success if p >= 0.5 else st.warning)(
                "Dự đoán: **ĐĂNG KÝ**" if p >= 0.5 else "Dự đoán: **KHÔNG đăng ký**")

    if (p_nb >= 0.5) != (p_dt >= 0.5):
        st.error("⚡ Hai mô hình **BẤT ĐỒNG**! Sang tab 🧮 và 🌳 để xem mỗi mô hình lý giải thế nào.")
    else:
        st.info("Hai mô hình cùng kết luận — nhưng chúng đi đến đó bằng hai con đường hoàn toàn khác nhau (xem 2 tab kế tiếp).")
    if truth is not None:
        st.markdown(f"**Đáp án thật của khách này:** {'✅ ĐÃ đăng ký' if truth == 1 else '❌ KHÔNG đăng ký'}")
    st.caption(f"Chỉ {vn_pct(nb.prior_[1])} khách đăng ký, nên rất ít khách vượt ngưỡng 50%. "
               "Vì vậy ở tab cuối ta dùng xác suất để **xếp hạng** khách hàng thay vì chỉ trả lời Có/Không.")

    st.markdown("""
| | 🧮 Naive Bayes | 🌳 Decision Tree |
|---|---|---|
| **Cách “nghĩ”** | Cộng dồn bằng chứng từ *mọi* thông tin | Hỏi lần lượt các câu Có/Không |
| **Kết quả** | Một xác suất có thể giải thích từng phần | Một đường đi + luật if-else dễ đọc |
| **Giả định** | Các thông tin *độc lập* với nhau | Không giả định gì |
| **Điểm yếu** | Thông tin trùng lặp bị “đếm hai lần” | Cây quá sâu sẽ học thuộc lòng (overfit) |
""")

# ================================================================== TAB 2 — NAIVE BAYES
with tab_nb:
    st.markdown("### Naive Bayes = xuất phát từ tỉ lệ nền, rồi để **từng thông tin “bỏ phiếu”**")
    odds0 = nb.prior_[1] / nb.prior_[0]
    st.markdown(
        f"**Bước 0 — Xác suất nền (prior).** Trong {vn_num(nb.n_[0] + nb.n_[1])} khách ở tập học, "
        f"{vn_num(nb.n_[1])} người đăng ký → **{vn_pct(nb.prior_[1])}**, tức odds ≈ 1 : {vn_num(1 / odds0, 1)}. "
        "Sau đó mỗi thông tin về khách hàng nhân odds với một **hệ số bằng chứng** "
        "= *P(thông tin | đăng ký) ÷ P(thông tin | không đăng ký)*.")

    n_feat = len(expl)
    k = st.slider("Kéo để xét thêm từng thông tin (thông tin mạnh nhất được xét trước)", 0, n_feat,
                  min(3, n_feat), key="nb_k")
    m1, m2 = st.columns([1, 2])
    with m1:
        prev = evo[k - 1] if k > 0 else evo[0]
        st.metric(f"P(đăng ký) sau khi xét {k}/{n_feat} thông tin", vn_pct(evo[k], 1),
                  delta=(f"{(evo[k] - prev) * 100:+.1f} điểm %".replace(".", ",") if k > 0 else "điểm xuất phát"))
        st.progress(float(evo[k]))
        st.caption("Nhân hết các hệ số ⇒ chính là con số Naive Bayes trả về ở tab 🎯." if k == n_feat else
                   "Kéo tiếp thanh trượt để thêm bằng chứng…")
    with m2:
        fig_out(core.fig_nb_votes(expl, k))

    if k > 0:
        tbl = expl.head(k).copy()
        show = pd.DataFrame({
            "Thông tin": [core.LABEL[f] for f in tbl["feature"]],
            "Khách hàng": [core.show_value(f, v, kd) for f, v, kd in zip(tbl["feature"], tbl["value"], tbl["kind"])],
            "P(… | ĐĂNG KÝ)": [vn_pct(p, 1) if kd == "cat" else f"{p:.5f}" for p, kd in zip(tbl["p_yes"], tbl["kind"])],
            "P(… | KHÔNG)": [vn_pct(p, 1) if kd == "cat" else f"{p:.5f}" for p, kd in zip(tbl["p_no"], tbl["kind"])],
            "Hệ số ×": [vn_num(float(np.exp(lr)), 2) for lr in tbl["log_ratio"]],
            "P(đăng ký) sau bước này": [vn_pct(p, 1) for p in evo[1:k + 1]],
        })
        st.dataframe(show, hide_index=True)
        st.caption("Với thông tin dạng số, P là *mật độ* của đường cong chuẩn (không phải xác suất), nhưng tỉ số vẫn dùng như nhau.")

    st.divider()
    st.markdown("#### 🔍 Nhìn sâu vào một thông tin: “P(giá trị | lớp)” từ đâu ra?")
    names = [core.LABEL[f] for f in expl["feature"]]
    pick = st.selectbox("Chọn thông tin", names, index=0, key="nb_pick")
    er = expl.iloc[names.index(pick)]
    d1, d2 = st.columns([3, 2])
    with d1:
        fig_out(core.fig_feature_detail(nb, er, tr, tr["y"]))
    with d2:
        ratio = float(np.exp(er["log_ratio"]))
        if er["kind"] == "cat":
            st.markdown(
                f"Trong **{vn_num(nb.n_[1])}** khách ĐÃ đăng ký, có **{vn_num(er['count_yes'])}** người thuộc nhóm "
                f"“{core.show_value(er['feature'], er['value'])}” → **{vn_pct(er['p_yes'], 1)}**.\n\n"
                f"Trong **{vn_num(nb.n_[0])}** khách KHÔNG đăng ký, có **{vn_num(er['count_no'])}** người → **{vn_pct(er['p_no'], 1)}**.\n\n"
                f"Tỉ số hai con số = **×{vn_num(ratio, 2)}** — đó là “lá phiếu” của thông tin này.")
            st.caption("Naive Bayes “học” chỉ bằng cách **đếm** như vậy (có cộng 1 làm mịn Laplace).")
        else:
            st.markdown(
                f"Khách ĐĂNG KÝ có {core.LABEL[er['feature']].lower()} trung bình **{vn_num(er['mean_yes'])}** "
                f"(độ lệch {vn_num(er['sd_yes'])}); khách KHÔNG đăng ký: **{vn_num(er['mean_no'])}** "
                f"(độ lệch {vn_num(er['sd_no'])}).\n\n"
                f"Tại giá trị **{int(er['value'])}**, đường cong ĐĂNG KÝ cao {er['p_yes']:.5f}, đường cong KHÔNG cao {er['p_no']:.5f} "
                f"→ hệ số **×{vn_num(ratio, 2)}**.")
            st.caption("Cột mờ là dữ liệu thật — thường lệch phải, không phải hình chuông. NB vẫn dùng được vì chỉ cần "
                       "so sánh hai đường cong; nhưng đây là điểm “ngây thơ” của giả định Gaussian.")

    st.divider()
    st.markdown("#### 🧪 Thí nghiệm: vì sao gọi là “Naive” (ngây thơ)?")
    st.markdown("Naive Bayes **giả định các thông tin độc lập**, nên nếu cùng một thông tin xuất hiện nhiều lần "
                "(ví dụ hai cột nói cùng một điều) nó coi đó là nhiều bằng chứng mới và nhân hệ số nhiều lần → "
                "**quá tự tin**. Cây quyết định thì không bị ảnh hưởng.")
    cat_rows = expl[expl["kind"] == "cat"]
    cat_names = [core.LABEL[f] for f in cat_rows["feature"]]
    dup_pick = st.selectbox("Thông tin sẽ được lặp lại", cat_names, index=0, key="dup_pick")
    dup_feat = cat_rows["feature"].iloc[cat_names.index(dup_pick)]
    fig_out(core.fig_double_counting(nb, expl, dup_feat, p_dt))

# ================================================================== TAB 3 — DECISION TREE
with tab_dt:
    st.markdown("### Decision Tree = chuỗi câu hỏi **Có / Không** dẫn tới một lá")
    st.markdown("Cây tự học câu hỏi nào chia khách hàng thành hai nhóm “thuần” nhất. "
                "**Đường viền đỏ** là hành trình của khách hàng đang xét. "
                "Màu **cam** = nhóm nghiêng KHÔNG đăng ký, **xanh** = nghiêng ĐĂNG KÝ.")
    dot = core.tree_to_dot(DT, cols, [s["node"] for s in steps] + [leaf])
    show_tree(dot)

    st.markdown("#### Hành trình của khách hàng này")
    for i, s in enumerate(steps, 1):
        extra = f" *(khách hàng: {s['value']})*" if s["value"] else ""
        st.markdown(f"**Câu {i}.** {s['question']}{extra} → **{'✅ CÓ' if s['answer'] else '❌ KHÔNG'}**")
    (st.success if leaf_p >= 0.5 else st.info)(
        f"Đến lá cuối: trong **{vn_num(leaf_n)}** khách ở tập học giống khách này, **{vn_pct(leaf_p)}** đã đăng ký "
        f"→ dự đoán **{'ĐĂNG KÝ' if leaf_p >= 0.5 else 'KHÔNG đăng ký'}**.")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Cây quan tâm tới thông tin nào nhất?")
        fig_out(core.fig_importance(DT, cols))
    with e2:
        st.markdown("#### ⚠️ Vì sao không cho cây sâu mãi? (Overfitting)")
        curve = get_curve(N_ROWS, use_duration, B)
        fig_out(core.fig_overfit(curve, depth, 1 - float(te["y"].mean())))
        st.caption(f"Độ sâu {depth}: học {vn_pct(DT.score(B['Xtr'], tr['y']))} · test {vn_pct(DT.score(B['Xte'], te['y']))}. "
                   "Cây sâu “học thuộc lòng” tập học (đường nhạt lên cao) nhưng giảm ở dữ liệu mới.")

# ================================================================== TAB 4 — ỨNG DỤNG
with tab_biz:
    st.markdown("### Ứng dụng thực tế: đội telesales chỉ gọi được một phần danh sách — nên gọi ai trước?")
    st.markdown("Dùng **xác suất** mà mô hình cho ra để **xếp hạng** khách hàng, rồi gọi từ trên xuống. "
                "So với gọi ngẫu nhiên, mô hình nào tìm được nhiều người đăng ký hơn với cùng số cuộc gọi?")
    budget = st.slider("Đội ngũ chỉ gọi được bao nhiêu % danh sách?", 5, 100, 20, step=5, key="budget")

    y_te = te["y"].to_numpy()
    xn, gn = core.gains_curve(y_te, B["p_nb"])
    xd, gd = core.gains_curve(y_te, P_DT_TEST)
    pos_nb = float((B["p_nb"] > p_nb).mean() * 100)
    pos_dt = float((P_DT_TEST > p_dt).mean() * 100)

    g1, g2 = st.columns([3, 2])
    with g1:
        fig_out(core.fig_gains(xn, gn, xd, gd, budget, pos_nb, pos_dt))
        st.caption("▼ trên trục hoành: vị trí của khách hàng đang xét trong bảng xếp hạng (càng gần 0% càng được ưu tiên gọi).")
    with g2:
        n_call = int(len(te) * budget / 100)
        cap_nb, cap_dt = core.captured_at(xn, gn, budget), core.captured_at(xd, gd, budget)
        st.metric(f"Gọi {vn_num(n_call)} / {vn_num(len(te))} khách — Naive Bayes tìm được", f"{cap_nb:.0f}% người đăng ký".replace(".", ","))
        st.metric("Decision Tree tìm được", f"{cap_dt:.0f}% người đăng ký".replace(".", ","))
        st.metric("Gọi ngẫu nhiên tìm được", f"~{budget}%")
        best = max(cap_nb, cap_dt)
        if budget < 100:
            st.success(f"Hiệu quả gấp **~{vn_num(best / budget, 1)} lần** gọi ngẫu nhiên, cùng số cuộc gọi.")
        st.caption(f"Khách đang xét nằm ở top {vn_num(pos_nb, 0)}% (NB) và top {vn_num(pos_dt, 0)}% (DT) danh sách.")

    st.divider()
    st.markdown("#### Bảng chỉ số trên tập test (khách chưa từng thấy khi học)")
    mn, md = core.metrics_at_threshold(y_te, B["p_nb"]), core.metrics_at_threshold(y_te, P_DT_TEST)
    base = 1 - float(y_te.mean())
    st.dataframe(pd.DataFrame({
        "Chỉ số": ["Accuracy (đoán đúng)", "Precision (đoán ĐK thì đúng bao nhiêu)", "Recall (tìm được bao nhiêu người ĐK)", "AUC (khả năng xếp hạng)"],
        "Naive Bayes": [vn_pct(mn["accuracy"]), vn_pct(mn["precision"]), vn_pct(mn["recall"]), vn_num(mn["auc"], 3)],
        "Decision Tree": [vn_pct(md["accuracy"]), vn_pct(md["precision"]), vn_pct(md["recall"]), vn_num(md["auc"], 3)],
    }), hide_index=True)
    st.warning(f"⚠️ **Bẫy Accuracy:** một “mô hình lười” luôn đoán KHÔNG cũng đạt **{vn_pct(base)}**! "
               "Vì vậy phải nhìn thêm Recall và AUC — và đây là lý do xếp hạng (Gains) có ý nghĩa kinh doanh hơn.")
    if not use_duration:
        st.info("Bạn đang **bỏ duration**: mô hình yếu đi rõ rệt (AUC giảm) — cho thấy phần lớn “sức mạnh” trước đó "
                "đến từ một thông tin chỉ có SAU khi cuộc gọi kết thúc (rò rỉ thông tin).")
