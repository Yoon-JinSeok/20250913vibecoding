# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st

# --------------- 유틸 ---------------
def parse_answer_key(s):
    if s is None:
        return []
    s = s.strip()
    if "," in s:
        keys = [x.strip().upper() for x in s.split(",") if x.strip()]
    else:
        keys = list(s.upper())
    return keys

def detect_item_columns(df, pattern=r"^Q(\d+)$"):
    item_cols = []
    for c in df.columns:
        m = re.match(pattern, str(c).strip())
        if m:
            item_cols.append(c)
    # 숫자 기준 정렬
    def keyer(x):
        m = re.match(pattern, str(x).strip())
        return int(m.group(1)) if m else 10**9
    item_cols.sort(key=keyer)
    return item_cols

def kr20(binary_matrix):
    """
    binary_matrix: (n_students, k_items) 0/1 numpy array
    KR-20 = k/(k-1) * (1 - sum(p*q)/var(total))
    """
    if binary_matrix.size == 0:
        return np.nan
    k = binary_matrix.shape[1]
    if k < 2:
        return np.nan
    p = binary_matrix.mean(axis=0)
    q = 1 - p
    total = binary_matrix.sum(axis=1)
    var_total = np.var(total, ddof=1)  # sample variance
    if var_total == 0:
        return np.nan
    return (k/(k-1)) * (1 - (np.sum(p*q) / var_total))

def point_biserial(item_scores, total_scores_excl):
    """
    상관계수(r) 계산. item_scores(0/1)와 해당 문항 제외 총점의 피어슨 상관.
    """
    if np.std(item_scores, ddof=1) == 0 or np.std(total_scores_excl, ddof=1) == 0:
        return np.nan
    return np.corrcoef(item_scores, total_scores_excl)[0,1]

def make_binary_matrix(resp_df, keys, choices=("A","B","C","D","E")):
    """
    응답(A~E 등)과 정답키로 0/1 행렬 생성
    """
    k = len(keys)
    if k == 0:
        return None
    n = len(resp_df)
    bin_mat = np.zeros((n, k), dtype=float)
    for j in range(k):
        key = keys[j]
        col = resp_df.columns[j]
        if key == "" or key is None:
            bin_mat[:, j] = np.nan
            continue
        # 대문자 비교
        bin_mat[:, j] = (resp_df[col].astype(str).str.upper().values == key).astype(float)
    return bin_mat

def distractor_analysis(resp_df, keys, choice_set=("A","B","C","D","E")):
    """
    각 문항별 선택지 분포와 선택지-총점 상관(선택지 인디케이터 vs 총점)을 계산
    """
    results = {}
    # 먼저 전체 이분점수로 총점(정답 개수) 계산
    bin_mat = make_binary_matrix(resp_df, keys, choice_set)
    if bin_mat is None:
        return results
    total = np.nansum(bin_mat, axis=1)

    for j, col in enumerate(resp_df.columns):
        key = keys[j] if j < len(keys) else None
        series = resp_df[col].astype(str).str.upper()
        dist = series.value_counts(dropna=False).reindex(list(choice_set), fill_value=0)
        # 선택지별 상관
        corrs = {}
        for ch in choice_set:
            ind = (series == ch).astype(float).values
            # 총점 상관
            if np.std(ind, ddof=1) == 0 or np.std(total, ddof=1) == 0:
                corrs[ch] = np.nan
            else:
                corrs[ch] = float(np.corrcoef(ind, total)[0,1])
        results[col] = {
            "correct": key,
            "counts": dist.to_dict(),
            "corrs": corrs
        }
    return results

def export_csv_button(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

# --------------- 앱 시작 ---------------
st.set_page_config(page_title="학력평가 문항분석", layout="wide")
st.title("📊 학력평가 문항분석(CTT) 대시보드")

st.sidebar.header("1) 데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 또는 Excel 업로드", type=["csv","xlsx","xls"])
id_col = st.sidebar.text_input("학생 ID 컬럼명(선택)", value="student_id")
subject_col = st.sidebar.text_input("과목 컬럼명(선택)", value="subject")

st.sidebar.header("2) 문항/정답 설정")
auto_detect = st.sidebar.checkbox("문항 컬럼 자동 감지(Q1, Q2 ...)", value=True)
pattern = st.sidebar.text_input("자동 감지 패턴(정규식)", value=r"^Q(\d+)$")

answer_key_input = st.sidebar.text_area(
    "정답키 입력 (예: ABCDE... 또는 A,B,C,D,E ...)", height=100
)
choice_set_input = st.sidebar.text_input("선택지 집합(쉼표로, 대문자)", value="A,B,C,D,E")
choice_set = tuple([x.strip().upper() for x in choice_set_input.split(",") if x.strip()])

flag_easy = st.sidebar.slider("너무 쉬움 임계(p ≥)", 0.50, 1.00, 0.90, 0.01)
flag_hard = st.sidebar.slider("너무 어려움 임계(p ≤)", 0.00, 0.50, 0.20, 0.01)
flag_lowdisc = st.sidebar.slider("저변별 임계(r_pb ≤)", 0.00, 0.50, 0.20, 0.01)

if uploaded is None:
    st.info("좌측에서 파일을 업로드하면 분석을 시작합니다. 예: student_id, subject, Q1~Q50 ...")
    st.stop()

# 파일 로드
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded, dtype=str).fillna("")
else:
    df = pd.read_excel(uploaded, dtype=str).fillna("")

# 문항 컬럼 선택
if auto_detect:
    item_cols = detect_item_columns(df, pattern=pattern)
else:
    all_cols = list(df.columns)
    item_cols = st.multiselect("문항 컬럼 선택", all_cols, default=[c for c in all_cols if re.match(r"^Q\d+$", str(c))])

if len(item_cols) == 0:
    st.error("문항 컬럼을 찾지 못했습니다. 자동 패턴을 조정하거나 수동 선택하세요.")
    st.stop()

# 정답키 파싱
keys = parse_answer_key(answer_key_input)
if len(keys) not in (0, len(item_cols)):
    st.warning(f"정답 문항 수({len(keys)})가 선택한 문항 수({len(item_cols)})와 다릅니다. 정답키를 확인하세요.")
# 정답키 길이가 0이면 분석은 응답분포·기초 통계까지만 가능
keys = (keys if len(keys) == len(item_cols) else [""]*len(item_cols))

# 응답 데이터프레임(문항만)
resp_df = df[item_cols].copy()

# 대문자로 통일
for c in item_cols:
    resp_df[c] = resp_df[c].astype(str).str.upper().str.strip()

# 이분 점수 행렬
bin_mat = make_binary_matrix(resp_df, keys, choice_set)
if bin_mat is not None:
    total_scores = np.nansum(bin_mat, axis=1)
    # 문항 제외 총점 (변별도 계산용)
    excl_totals = []
    for j in range(len(item_cols)):
        excl_totals.append(total_scores - bin_mat[:, j])
    excl_totals = np.array(excl_totals).T  # (n, k)
else:
    total_scores = None
    excl_totals = None

# ----- 요약 카드 -----
left, mid, right = st.columns(3)
with left:
    st.metric("학생 수", f"{len(df):,}")
with mid:
    st.metric("문항 수", f"{len(item_cols)}")
with right:
    if bin_mat is not None:
        st.metric("KR-20(신뢰도)", f"{kr20(bin_mat):.3f}")
    else:
        st.metric("KR-20(신뢰도)", "정답키 필요")

# ----- 성적/분포 -----
st.subheader("📈 점수 분포")
if bin_mat is not None:
    score_df = pd.DataFrame({
        "student_id": df[id_col] if id_col in df.columns else np.arange(len(df))+1,
        "subject": df[subject_col] if subject_col in df.columns else "",
        "total_score": total_scores,
        "percent": (total_scores/len(item_cols))*100
    })
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.bar_chart(score_df["total_score"])
    export_csv_button(score_df, "점수표 CSV 다운로드", "scores.csv")
else:
    st.info("정답키가 있어야 총점/분포를 계산합니다.")

# ----- 문항 분석표 -----
st.subheader("🧪 문항 통계(난이도·변별도)")
item_stats = []
if bin_mat is not None:
    for j, col in enumerate(item_cols):
        item_bin = bin_mat[:, j]
        # 난이도 p
        valid = ~np.isnan(item_bin)
        p = item_bin[valid].mean() if valid.any() else np.nan
        # 변별도 (해당 문항 제외 총점)
        rpb = point_biserial(item_bin[valid], excl_totals[valid, j]) if valid.any() else np.nan
        flags = []
        if not np.isnan(p) and p >= flag_easy: flags.append("너무 쉬움")
        if not np.isnan(p) and p <= flag_hard: flags.append("너무 어려움")
        if not np.isnan(rpb) and rpb <= flag_lowdisc: flags.append("저변별")
        item_stats.append({
            "item": col,
            "key": keys[j],
            "p(정답률)": round(float(p), 3) if p==p else np.nan,
            "r_pb(변별도)": round(float(rpb), 3) if rpb==rpb else np.nan,
            "flag": ", ".join(flags)
        })
    item_stats_df = pd.DataFrame(item_stats)
    st.dataframe(item_stats_df, use_container_width=True, hide_index=True)
    export_csv_button(item_stats_df, "문항 통계 CSV 다운로드", "item_stats.csv")
else:
    st.info("정답키가 있어야 문항 난이도/변별도를 계산합니다.")

# ----- 오답지(선택지) 분석 -----
st.subheader("🔎 선택지(오답지) 분석")
if bin_mat is not None:
    dist = distractor_analysis(resp_df, keys, choice_set)
    sel_item = st.selectbox("문항 선택", item_cols)
    if sel_item in dist:
        info = dist[sel_item]
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"정답: **{info['correct']}**")
            counts_df = pd.DataFrame({
                "choice": list(info["counts"].keys()),
                "count": list(info["counts"].values())
            })
            st.bar_chart(counts_df.set_index("choice"))
            st.dataframe(counts_df, use_container_width=True, hide_index=True)
        with col2:
            corrs_df = pd.DataFrame({
                "choice": list(info["corrs"].keys()),
                "corr_with_total": list(info["corrs"].values())
            })
            st.dataframe(corrs_df, use_container_width=True, hide_index=True)
    # 전체 요약 CSV(선택지 분포·상관 펼치기)
    flat_rows = []
    for it, info in dist.items():
        for ch in choice_set:
            flat_rows.append({
                "item": it,
                "choice": ch,
                "is_key": (ch == info["correct"]),
                "count": info["counts"].get(ch, 0),
                "corr_with_total": info["corrs"].get(ch, np.nan)
            })
    dist_df = pd.DataFrame(flat_rows)
    export_csv_button(dist_df, "선택지 분석 CSV 다운로드", "distractor_analysis.csv")
else:
    st.info("정답키가 있어야 오답지 분석을 볼 수 있습니다.")

st.divider()
st.caption("Tip) 영역/성취기준 태그 컬럼을 데이터에 추가하면 영역별 통계도 쉽게 만들 수 있어요(피벗테이블).")
