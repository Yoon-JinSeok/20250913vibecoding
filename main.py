# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------- 공통 유틸 ----------------
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
    def keyer(x):
        m = re.match(pattern, str(x).strip())
        return int(m.group(1)) if m else 10**9
    item_cols.sort(key=keyer)
    return item_cols

def kr20(binary_matrix):
    """
    binary_matrix: (n_students, k_items), values in {0,1}
    KR-20 = k/(k-1) * (1 - sum(p*q)/var(total))
    """
    if binary_matrix is None or binary_matrix.size == 0:
        return np.nan
    k = binary_matrix.shape[1]
    if k < 2:
        return np.nan
    p = np.nanmean(binary_matrix, axis=0)
    q = 1 - p
    total = np.nansum(binary_matrix, axis=1)
    var_total = np.var(total, ddof=1)
    if var_total == 0:
        return np.nan
    return (k/(k-1)) * (1 - (np.nansum(p*q) / var_total))

def point_biserial(item_scores, total_scores_excl):
    if np.std(item_scores, ddof=1) == 0 or np.std(total_scores_excl, ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(item_scores, total_scores_excl)[0,1])

def make_binary_matrix(resp_df, keys, choices=("A","B","C","D","E")):
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
        bin_mat[:, j] = (resp_df[col].astype(str).str.upper().values == key).astype(float)
    return bin_mat

def distractor_analysis(resp_df, keys, choice_set=("A","B","C","D","E")):
    results = {}
    bin_mat = make_binary_matrix(resp_df, keys, choice_set)
    if bin_mat is None:
        return results
    total = np.nansum(bin_mat, axis=1)

    for j, col in enumerate(resp_df.columns):
        key = keys[j] if j < len(keys) else None
        series = resp_df[col].astype(str).str.upper()
        dist = series.value_counts(dropna=False).reindex(list(choice_set), fill_value=0)
        corrs = {}
        for ch in choice_set:
            ind = (series == ch).astype(float).values
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

def make_sample_template(n_items=20, choice_set=("A","B","C","D","E")):
    cols = ["student_id","grade","class","subject","test_name","taken_at"]
    cols += [f"Q{i}" for i in range(1, n_items+1)]
    data = []
    for sid in range(1, 6):
        row = [f"S{sid:03d}", 2, 1, "수학", "3월 학력평가", "2025-03-07"]
        row += [np.random.choice(choice_set) for _ in range(n_items)]
        data.append(row)
    df = pd.DataFrame(data, columns=cols)
    return df

# ---------------- 앱 설정 ----------------
st.set_page_config(page_title="학력평가 문항분석(CTT) 대시보드", layout="wide")
st.title("📊 학력평가 문항분석(CTT) 대시보드")

tab_dash, tab_guide = st.tabs(["분석 대시보드", "가이드/설명"])

# =========================================================
# 탭 1) 분석 대시보드
# =========================================================
with tab_dash:
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

    st.sidebar.markdown("---")
    st.sidebar.caption("샘플 템플릿")
    n_items = st.sidebar.number_input("샘플 문항 수", 5, 50, 20, 1)
    if st.sidebar.button("샘플 CSV 다운로드"):
        sample = make_sample_template(n_items=n_items, choice_set=choice_set)
        export_csv_button(sample, "샘플 CSV 저장", f"sample_{n_items}items.csv")

    if uploaded is None:
        st.info("좌측에서 파일을 업로드하면 분석을 시작합니다. 예: student_id, subject, Q1~Qn ...")
        st.stop()

    # ----- 파일 로드 -----
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str).fillna("")
    else:
        df = pd.read_excel(uploaded, dtype=str).fillna("")

    # ----- 문항 컬럼 선택 -----
    if auto_detect:
        item_cols = detect_item_columns(df, pattern=pattern)
    else:
        all_cols = list(df.columns)
        default_items = [c for c in all_cols if re.match(r"^Q\d+$", str(c))]
        item_cols = st.multiselect("문항 컬럼 선택", all_cols, default=default_items)

    if len(item_cols) == 0:
        st.error("문항 컬럼을 찾지 못했습니다. 자동 패턴을 조정하거나 수동 선택하세요.")
        st.stop()

    # ----- 정답키 -----
    keys = parse_answer_key(answer_key_input)
    if len(keys) not in (0, len(item_cols)):
        st.warning(f"정답 문항 수({len(keys)})가 선택한 문항 수({len(item_cols)})와 다릅니다. 정답키를 확인하세요.")
    keys = (keys if len(keys) == len(item_cols) else [""]*len(item_cols))

    # ----- 응답 데이터프레임(문항만) -----
    resp_df = df[item_cols].copy()
    for c in item_cols:
        resp_df[c] = resp_df[c].astype(str).str.upper().str.strip()

    # ----- 이분 점수 행렬/총점 -----
    bin_mat = make_binary_matrix(resp_df, keys, choice_set)
    if bin_mat is not None:
        total_scores = np.nansum(bin_mat, axis=1)
        excl_totals = []
        for j in range(len(item_cols)):
            excl_totals.append(total_scores - bin_mat[:, j])
        excl_totals = np.array(excl_totals).T
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

    # ----- 점수 분포 -----
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

    # ----- 문항 통계 -----
    st.subheader("🧪 문항 통계(난이도·변별도)")
    if bin_mat is not None:
        item_stats = []
        for j, col in enumerate(item_cols):
            item_bin = bin_mat[:, j]
            valid = ~np.isnan(item_bin)
            p = item_bin[valid].mean() if valid.any() else np.nan
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
        # 펼친 요약 CSV
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

# =========================================================
# 탭 2) 가이드/설명 (요청하신 1·2번 방법론 표시)
# =========================================================
with tab_guide:
    st.header("1) 고등학교 학력평가 데이터를 다운받는 법")
    st.markdown(
        """
**A. NEIS 성적/평가 관리에서 내보내기(학교 자체 처리)**  
- 경로 예시: 성적/평가 관리 → 학력평가(또는 ‘모의고사/학력평가’) → 성적/문항반응 자료 → **엑셀/CSV 내보내기**  
- 포함 권장 항목  
  - 익명 ID(학번→별도 매핑), 학년/반, 시험명·시행일  
  - 과목/영역, **문항별 응답(Q1~Qn: A~E)**, (가능하면) 문항 정답·배점  
- 개인정보보호  
  - 실명/주민번호/전화 등은 제외, 학번→익명ID로 치환 후 반출

**B. OMR 스캐너/업체(채점 프로그램)에서 내보내기**  
- 채점 SW에서 “문항반응 데이터(원자료)” 또는 “학생별 원점수/문항별 응답”을 **CSV/Excel**로 내보내기  
- 옵션에 **학생식별자(익명), 문항별 응답, 정답/배점, 과목/영역** 포함

**C. 자체 시스템/스프레드시트 사용**  
- 스프레드시트로 관리 중이면 **CSV 다운로드**  
- 형식 통일(권장 컬럼):  
  - `student_id`(익명), `grade`, `class`, `subject`, `test_name`, `taken_at`  
  - `Q1`…`Qn` (값: A~E). 정답키는 파일 외부/앱 입력으로 별도 관리 권장
        """
    )
    with st.expander("권장 컬럼 표준 & 샘플 템플릿"):
        st.code(
            "student_id,grade,class,subject,test_name,taken_at,Q1,Q2,...,Qn\n"
            "S001,2,1,수학,3월 학력평가,2025-03-07,A,C,D,B,...\n"
            "S002,2,1,수학,3월 학력평가,2025-03-07,B,B,D,E,...",
            language="text",
        )
        st.caption("좌측 사이드바에서 ‘샘플 CSV 다운로드’ 버튼으로 예시 데이터를 내려받을 수 있어요.")

    st.header("2) 문항분석 방법론(CTT) 요약")
    st.markdown(
        """
**(1) 난이도 p (정답률)**  
- 정의: 문항 정답 비율(0~1).  
- 해석 가이드: 보통 **0.3~0.8** 적정, **0.2↓** 너무 어려움, **0.9↑** 너무 쉬움.  

**(2) 변별도 r_pb (점-이변량 상관; point-biserial)**  
- 정의: 각 문항 득점(0/1)과 **해당 문항을 제외한 총점** 간 상관.  
- 해석 가이드: **0.2↑ 권장**, **0.1~0.2** 개선 검토, **0.1↓** 교체 검토.  

**(3) 신뢰도 KR-20 / Cronbach’s α**  
- 정의: 내적일관성(문항들이 같은 구인을 얼마나 일관되게 측정하는가).  
- 가이드: **0.7↑ 권장**(문항 수·동질성·시험 목적에 따라 달라질 수 있음).  

**(4) 선택지(오답지) 분석**  
- 각 선택지 응답 **분포**(A~E) 및 **선택지-총점 상관**을 점검.  
  - 정답 선택지는 총점과 **양(+)의 상관**이 기대되고, 오답 선택지는 **음수에 가까운 상관**이 이상적.  
  - 특정 오답에 고득점자가 몰리면 **미끼 기능 약함** 또는 **문항 오해 요소** 가능성.  

**(5) 품질 체크 기준(현장 예시)**  
- 난이도 p: **0.25~0.85** 권장  
- 변별도 r_pb: **0.2 이상 권장**, 0.1~0.2 개선 검토, 0.1 미만 교체 검토  
- KR-20: **0.7 이상 권장**  
        """
    )
    st.info("본 대시보드는 상기 지표를 자동 계산/표시합니다. 좌측에서 정답키와 임계값을 조절해 보세요.")

    st.subheader("운영 팁")
    st.markdown(
        """
- **응답 표준화**: 모두 대문자 A~E, 무응답은 빈칸/NA  
- **정답키 관리**: 파일 컬럼보다 앱 입력/별도 CSV 권장(보안·수정 용이)  
- **폴더링**: `시험명/시행일/학년/과목` 규칙으로 파일명 관리  
- **익명화**: 외부 반출 전 실명 제거·익명 ID만 유지  
        """
    )

    st.caption(f"문서 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

