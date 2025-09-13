# app.py
import io
import os
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="MBTI 국가 Top 10", layout="wide")
st.title("MBTI 유형별 비율이 가장 높은 국가 Top 10")

# ---------------------------
# 1) CSV 파일 우선 읽기 (같은 폴더)
# ---------------------------
DEFAULT_FILE = "countriesMBTI_16types.csv"

def load_csv_with_enc(path_or_buf):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path_or_buf, encoding=enc), enc
        except Exception:
            continue
    return pd.read_csv(path_or_buf), "unknown"

df, used_encoding = None, None

if os.path.exists(DEFAULT_FILE):
    df, used_encoding = load_csv_with_enc(DEFAULT_FILE)
    st.success(f"기본 파일 `{DEFAULT_FILE}` (encoding={used_encoding}) 로드 완료")
else:
    st.info("기본 CSV 파일이 폴더에 없습니다. 업로드한 파일을 사용하세요.")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is not None:
        raw = uploaded.read()
        df, used_encoding = load_csv_with_enc(io.BytesIO(raw))
        st.success(f"업로드 파일 로드 완료 (encoding={used_encoding})")
    else:
        st.stop()

# ---------------------------
# 2) 국가/지역 컬럼 & MBTI 컬럼 추정
# ---------------------------
MBTI_TYPES = {
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
}

obj_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
if len(obj_cols) == 1:
    id_col = obj_cols[0]
elif len(obj_cols) > 1:
    id_col = st.selectbox("국가/지역 컬럼 선택", obj_cols)
else:
    df = df.copy()
    df["ID"] = np.arange(len(df))
    id_col = "ID"

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
mbti_cols = [c for c in numeric_cols if c in MBTI_TYPES]

if not mbti_cols:
    st.warning("MBTI 표준명과 일치하는 열이 없어 수치형 전체를 후보로 제시합니다.")
    mbti_cols = st.multiselect("MBTI 열 선택", numeric_cols, default=numeric_cols)
else:
    mbti_cols = st.multiselect("MBTI 열 선택", mbti_cols, default=mbti_cols)

if not mbti_cols:
    st.error("MBTI 열을 1개 이상 선택하세요.")
    st.stop()

# ---------------------------
# 3) 값 스케일 보정 (0~100 → 0~1)
# ---------------------------
max_val = df[mbti_cols].to_numpy(np.float64).max()
if max_val > 1.2:
    df[mbti_cols] = df[mbti_cols] / 100.0
    st.caption("퍼센트 값으로 감지되어 100으로 나눠 비율로 변환했습니다.")

# ---------------------------
# 4) UI: MBTI 유형 선택, Top-N
# ---------------------------
col1, col2 = st.columns([2,1])
with col1:
    selected_type = st.selectbox("MBTI 유형 선택", sorted(mbti_cols))
with col2:
    top_n = st.number_input("Top N", min_value=3, max_value=30, value=10)

# ---------------------------
# 5) Top N 국가 추출 및 시각화
# ---------------------------
df_sorted = df.sort_values(by=selected_type, ascending=False).head(int(top_n))
plot_df = df_sorted[[id_col, selected_type]].rename(columns={id_col: "country", selected_type: "value"})

highlight = alt.selection_point(on="mouseover", fields=["country"])
base = alt.Chart(plot_df).encode(
    y=alt.Y("country:N", sort="-x"),
    x=alt.X("value:Q", axis=alt.Axis(format="%")),
    tooltip=[alt.Tooltip("country:N"), alt.Tooltip("value:Q", format=".2%")]
)

bars = base.mark_bar().encode(
    color=alt.condition(highlight, alt.value("#1f77b4"), alt.value("#dddddd"))
).add_params(highlight)

text = base.mark_text(dx=5, align="left").encode(
    text=alt.Text("value:Q", format=".1%")
)

chart = (bars + text).properties(
    title=f"{selected_type} 비율 상위 {int(top_n)} 국가",
    width=750,
    height=30*len(plot_df)
).interactive()

st.altair_chart(chart, use_container_width=True)

# ---------------------------
# 6) 데이터 미리보기
# ---------------------------
with st.expander("데이터 미리보기"):
    st.dataframe(df.head(20))
