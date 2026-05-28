"""
BusAlert — 경기도 버스 도착·혼잡·이동 리스크 AI 안내 MVP

2026년 경기도 공공데이터·AI 활용 창업경진대회 아이디어 기획 부문 제출용
Streamlit MVP입니다. 기준 데이터는 저장소의 샘플 CSV를 사용하며, API 키가 없는
환경에서도 mock AI 분석으로 실행됩니다.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import os
from dataclasses import dataclass
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from streamlit_folium import st_folium


# ─────────────────────────────────────────────
# 0. 앱 설정값
# ─────────────────────────────────────────────
APP_TITLE = "BusAlert"
APP_SUBTITLE = "교통 취약지역 이용자를 위한 AI 기반 버스 도착·혼잡·이동 리스크 안내 MVP"
DATA_PATH = Path("최종_마스터_테이블.csv")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_WEIGHTS = {
    "demand": 0.40,
    "cv": 0.35,
    "accessibility": 0.15,
    "weather": 0.10,
}
REQUIRED_COLUMNS = {
    "노선ID",
    "노선명",
    "정류장순서",
    "station_name",
    "latitude",
    "longitude",
    "mean",
    "cv_score",
    "risk_score",
}
PUBLIC_DATA_SOURCES = [
    "경기데이터드림: 경기도 버스 정류소·노선·운행 관련 데이터 후보",
    "공공데이터포털: 버스 도착정보, 노선정보, 정류소정보 API 후보",
    "경기도 버스정보시스템/BMS: 실시간 위치·도착·배차 간격 연계 후보",
    "행정동·인구·교통약자 데이터: 고령자·학생·의료시설 접근성 보강 후보",
]
USER_TYPES = ["일반 이용자", "고령자", "학생", "보호자", "대중교통 정책 담당자"]
TIME_BANDS = ["출근 시간대", "낮 시간대", "퇴근 시간대", "야간 시간대", "비/눈 등 악천후"]
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]


@dataclass
class SidebarInputs:
    """사이드바 입력값 묶음."""

    weights: dict[str, float]
    top_n: int
    min_passenger: int
    growth_rate: int
    months_ahead: int
    user_type: str
    time_band: str
    selected_route_name: str | None
    concern: str
    use_real_llm: bool


# ─────────────────────────────────────────────
# 1. 기본 UI/스타일
# ─────────────────────────────────────────────
def configure_page() -> None:
    st.set_page_config(
        page_title="BusAlert — 경기도 버스 AI 리스크 안내",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 750; }
        .hero {background:linear-gradient(90deg,#172554,#1e3a8a);padding:22px 28px;border-radius:16px;margin-bottom:1.2rem;}
        .hero-eyebrow {color:#bfdbfe;font-size:.8rem;font-weight:800;letter-spacing:1.8px;margin-bottom:5px;}
        .hero-title {color:white;font-size:1.75rem;font-weight:850;}
        .hero-subtitle {color:#dbeafe;font-size:.95rem;margin-top:6px;}
        .section-title {font-size:1.08rem;font-weight:800;color:#0f172a;margin:1rem 0 .45rem;}
        .callout {border-left:5px solid #2563eb;background:#eff6ff;padding:12px 14px;border-radius:10px;margin:.5rem 0;}
        .risk-high {border-left:5px solid #dc2626;background:#fef2f2;padding:12px 14px;border-radius:10px;}
        .risk-mid {border-left:5px solid #f59e0b;background:#fffbeb;padding:12px 14px;border-radius:10px;}
        .risk-low {border-left:5px solid #16a34a;background:#f0fdf4;padding:12px 14px;border-radius:10px;}
        .small-muted {color:#64748b;font-size:.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 2. 데이터 로딩/검증
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="샘플 공공데이터를 불러오는 중입니다...")
def load_bus_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """CSV 샘플 데이터를 로딩하고 MVP 실행에 필요한 컬럼/좌표를 검증한다."""
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(sorted(missing))}")

    numeric_columns = ["정류장순서", "latitude", "longitude", "mean", "cv_score", "risk_score"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "mean", "cv_score", "risk_score"]).copy()
    df = df[
        (df["latitude"].between(33.0, 39.0))
        & (df["longitude"].between(124.0, 131.0))
    ].copy()
    df["노선명"] = df["노선명"].astype(str)
    df["station_name"] = df["station_name"].astype(str)
    return df.sort_values(["노선ID", "정류장순서", "station_name"]).reset_index(drop=True)


def safe_load_data() -> pd.DataFrame:
    """Streamlit 화면에서 데이터 오류를 사용자 친화적으로 보여준다."""
    try:
        df = load_bus_data()
    except Exception as exc:
        st.error(f"앱 실행에 필요한 데이터 로딩에 실패했습니다: {exc}")
        st.info("CSV 파일과 README의 실행 방법을 확인한 뒤 다시 실행해주세요.")
        st.stop()
    if df.empty:
        st.error("유효한 버스 정류장/노선 데이터가 없습니다.")
        st.stop()
    return df


# ─────────────────────────────────────────────
# 3. 분석/AI 로직
# ─────────────────────────────────────────────
def normalize(series: pd.Series) -> pd.Series:
    value_range = series.max() - series.min()
    if pd.isna(value_range) or value_range <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / value_range


def grade_priority(score: float) -> str:
    if score >= 70:
        return "🔴 최우선"
    if score >= 50:
        return "🟠 우선"
    if score >= 30:
        return "🟡 검토"
    return "🟢 관찰"


def compute_route_priority(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """노선별 수요·배차불균형·접근성 프록시를 결합해 안내 우선순위를 계산한다."""
    route_df = df.groupby(["노선ID", "노선명"], as_index=False).agg(
        start_station=("station_name", "first"),
        end_station=("station_name", "last"),
        boarding_count=("mean", "mean"),
        cv_score=("cv_score", "mean"),
        risk_score=("risk_score", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        stop_count=("정류장순서", "max"),
    )
    route_df["stop_count"] = route_df["stop_count"].clip(lower=1)
    route_df["accessibility_gap"] = normalize(route_df["stop_count"]) * 0.6 + normalize(route_df["cv_score"]) * 0.4
    route_df["weather_sensitivity"] = normalize(route_df["cv_score"] * route_df["stop_count"])
    route_df["priority_score"] = (
        normalize(route_df["boarding_count"]) * weights["demand"]
        + normalize(route_df["cv_score"]) * weights["cv"]
        + route_df["accessibility_gap"] * weights["accessibility"]
        + route_df["weather_sensitivity"] * weights["weather"]
    ).clip(0, 1)
    route_df["운행계통"] = route_df["start_station"] + " ↔ " + route_df["end_station"]
    route_df["예상최대대기분"] = (route_df["cv_score"] * 30 + 10).round(1)
    route_df["시간당운행대수"] = (15 - route_df["risk_score"] * 10).clip(lower=1).round(1)
    route_df["정보접근취약점수"] = (route_df["accessibility_gap"] * 100).round(1)
    route_df["종합위험점수"] = (route_df["priority_score"] * 100).round(1)
    route_df["등급"] = route_df["종합위험점수"].apply(grade_priority)
    return route_df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def predict_future_risk(route_df: pd.DataFrame, growth_rate_pct: int, months_ahead: int) -> pd.DataFrame:
    """수요 증가 시나리오를 적용해 미래 혼잡·이동 불편 위험을 시뮬레이션한다."""
    future = route_df.copy()
    if future.empty:
        return future
    growth = (1 + growth_rate_pct / 100) ** (months_ahead / 12)
    future["future_demand"] = future["boarding_count"] * growth
    future["future_cv"] = future["cv_score"] * (1 + 0.15 * np.log1p(max(growth - 1, 0)))
    future["future_risk"] = (
        normalize(future["future_demand"]) * 0.45
        + normalize(future["future_cv"]) * 0.40
        + normalize(future["accessibility_gap"]) * 0.15
    ).clip(0, 1)
    future["현재위험점수"] = (future["risk_score"] * 100).round(1)
    future["미래위험점수"] = (future["future_risk"] * 100).round(1)
    future["위험도변화"] = (future["미래위험점수"] - future["현재위험점수"]).round(1)
    future["예측시나리오"] = f"연 {growth_rate_pct}% 수요 증가, {months_ahead}개월 후"
    return future


@st.cache_data(show_spinner=False)
def train_risk_model(df: pd.DataFrame) -> tuple[float | None, dict[str, float], int, str | None]:
    """샘플 데이터 기반 ML 모델을 학습해 위험도 설명 피처를 산출한다."""
    route_agg = df.groupby("노선ID").agg(
        mean_pass=("mean", "mean"),
        cv_score=("cv_score", "mean"),
        stop_count=("정류장순서", "max"),
        lat_std=("latitude", "std"),
        risk_score=("risk_score", "mean"),
    ).dropna()
    if len(route_agg) < 3:
        return None, {}, len(route_agg), "모델 학습에는 최소 3개 이상 노선 샘플이 필요합니다."

    feature_columns = ["mean_pass", "cv_score", "stop_count", "lat_std"]
    model = GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=42)
    model.fit(route_agg[feature_columns], route_agg["risk_score"])
    train_r2 = float(model.score(route_agg[feature_columns], route_agg["risk_score"]))
    importances = dict(zip(feature_columns, model.feature_importances_.round(3)))
    return train_r2, importances, len(route_agg), None


def selected_route_or_default(route_df: pd.DataFrame, selected_route_name: str | None) -> pd.Series | None:
    if route_df.empty:
        return None
    if selected_route_name:
        matched = route_df[route_df["노선명"] == selected_route_name]
        if not matched.empty:
            return matched.iloc[0]
    return route_df.iloc[0]


def build_route_context(route_df: pd.DataFrame) -> str:
    rows = []
    for _, row in route_df.head(8).iterrows():
        rows.append(
            f"- {row['노선명']}번({row['운행계통']}): 평균 이용객 {row['boarding_count']:.0f}, "
            f"CV {row['cv_score']:.3f}, 정보접근취약 {row['정보접근취약점수']:.1f}, "
            f"종합위험 {row['종합위험점수']:.1f}, 등급 {row['등급']}"
        )
    return "\n".join(rows)


def mock_ai_analysis(route: pd.Series, user_type: str, time_band: str, concern: str) -> dict[str, str]:
    """API 키가 없어도 동작하는 규칙 기반 AI 분석 대체 함수."""
    score = float(route["종합위험점수"])
    wait = float(route["예상최대대기분"])
    cv = float(route["cv_score"])
    route_name = route["노선명"]

    if score >= 70:
        risk_label = "높음"
        action = "출발 전 알림을 2회로 설정하고, 동일 방향 대체 노선 또는 보호자 공유를 권장합니다."
        reason = "수요와 배차 변동성이 동시에 높아 정류장에서 체감 대기시간이 급격히 늘 수 있습니다."
    elif score >= 50:
        risk_label = "주의"
        action = "예상 도착 10분 전 알림과 혼잡 시간대 회피 안내를 우선 제공합니다."
        reason = "특정 시간대에 배차 간격이 벌어질 가능성이 있어 이동 지연 안내가 필요합니다."
    elif score >= 30:
        risk_label = "관찰"
        action = "평상시 안내는 유지하되 비·눈, 야간에는 주의 알림을 강화합니다."
        reason = "현재 위험은 중간 수준이나 접근성 취약 프록시와 기상 민감도를 함께 관찰해야 합니다."
    else:
        risk_label = "낮음"
        action = "기본 도착 알림과 정류장 위치 안내 중심으로 제공합니다."
        reason = "현 샘플 기준 혼잡·배차 불균형 신호가 상대적으로 낮습니다."

    persona_tip = {
        "고령자": "큰 글씨·짧은 문장으로 ‘지금 출발/조금 대기’를 명확히 안내합니다.",
        "학생": "등하교 시간 지각 위험과 혼잡 회피 출발 시점을 함께 제공합니다.",
        "보호자": "이동 리스크가 높을 때 보호자 공유 알림을 우선 추천합니다.",
        "대중교통 정책 담당자": "증차 후보, 배차 간격 조정, 정류장 안내 인프라 개선 근거로 활용합니다.",
    }.get(user_type, "초행자도 이해할 수 있도록 도착·대기·혼잡을 한 문장으로 요약합니다.")

    concern_text = f" 사용자가 입력한 우려사항은 ‘{concern}’입니다." if concern.strip() else ""
    summary = (
        f"{route_name}번 노선은 {time_band} 기준 이동 리스크가 ‘{risk_label}’입니다. "
        f"예상 최대 대기시간은 약 {wait:.1f}분, 배차 변동계수(CV)는 {cv:.3f}입니다.{concern_text}"
    )
    return {
        "risk_label": risk_label,
        "summary": summary,
        "reason": reason,
        "recommendation": action,
        "persona_tip": persona_tip,
        "judge_summary": (
            "이 MVP는 공공 버스 정류소·노선 샘플 데이터의 수요, CV, 위치 정보를 결합하고 "
            "AI 설명 계층을 얹어 취약 이용자에게 이해 가능한 이동 리스크와 알림 우선순위를 제공합니다."
        ),
    }


def ask_real_llm(question: str, route_context: str) -> str:
    """선택적으로 실제 LLM을 호출한다. 키가 없거나 패키지가 없으면 mock 경로를 사용한다."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
    if importlib.util.find_spec("anthropic") is None:
        raise RuntimeError("anthropic 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요.")

    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = f"""
당신은 경기도 버스정보 접근 취약지역을 지원하는 교통 AI 어시스턴트입니다.
아래 노선 데이터를 근거로 답변하고, 모르는 내용은 추정이라고 표시하세요.

{route_context}
"""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# 4. 화면 렌더링 함수
# ─────────────────────────────────────────────
def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">GYEONGGI PUBLIC DATA × AI MVP</div>
          <div class="hero-title">🚌 {APP_TITLE} — 경기도 버스 이동 리스크 안내</div>
          <div class="hero-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_problem_and_data() -> None:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">1) 해결하려는 문제</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="callout">
            교통 취약지역의 정류장 이용자는 단순 도착 예정시간만으로는 <b>긴 대기, 혼잡, 배차 불균형, 야간 이동 불안</b>을 판단하기 어렵습니다.
            BusAlert는 노선별 수요·CV·정류장 위치 데이터를 결합해 사용자가 이해하기 쉬운 알림 우선순위를 제공합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-title">2) 활용 공공데이터</div>', unsafe_allow_html=True)
        for source in PUBLIC_DATA_SOURCES:
            st.markdown(f"- {source}")
        st.caption("현재 MVP는 저장소 CSV 샘플 데이터로 데모하며, 실제 서비스에서는 API 연동 함수 위치를 교체합니다.")


def render_sidebar(df: pd.DataFrame) -> SidebarInputs:
    route_names = sorted(df["노선명"].astype(str).unique().tolist())
    with st.sidebar:
        st.markdown("## ⚙️ 데모 제어판")
        st.caption("심사위원이 가중치와 사용자 상황을 바꿔 즉시 결과를 확인할 수 있습니다.")

        st.markdown("### 사용자 입력 영역")
        selected_route_name = st.selectbox("관심 노선", ["자동: 위험도 상위 노선"] + route_names)
        user_type = st.selectbox("사용자 유형", USER_TYPES)
        time_band = st.selectbox("이용 시간/상황", TIME_BANDS)
        concern = st.text_area("사용자 우려사항", placeholder="예: 비 오는 날 병원 예약 시간에 늦지 않을지 걱정됩니다.")

        st.markdown("### 분석 가중치")
        weights = {
            "demand": st.slider("수요량", 0.0, 1.0, DEFAULT_WEIGHTS["demand"], 0.05),
            "cv": st.slider("배차 불균형(CV)", 0.0, 1.0, DEFAULT_WEIGHTS["cv"], 0.05),
            "accessibility": st.slider("교통/정보 접근 취약성", 0.0, 1.0, DEFAULT_WEIGHTS["accessibility"], 0.05),
            "weather": st.slider("기상·상황 민감도", 0.0, 1.0, DEFAULT_WEIGHTS["weather"], 0.05),
        }
        total_weight = sum(weights.values())
        st.caption(f"가중치 합계: {total_weight:.2f} {'✅' if abs(total_weight - 1.0) < 0.01 else '⚠️ 1.0 권장'}")

        st.markdown("### 표시/예측 설정")
        max_routes = max(1, df["노선ID"].nunique())
        top_n = st.slider("표시 노선 수", 1, max_routes, min(max_routes, 5))
        min_passenger = st.slider("최소 평균 이용객", 0, 1000, 0, 50)
        growth_rate = st.slider("연간 수요 증가율(%)", 0, 30, 10, 1)
        months_ahead = st.slider("예측 기간(개월)", 3, 24, 6, 3)
        use_real_llm = st.toggle("실제 LLM API 사용", value=False, help="ANTHROPIC_API_KEY가 설정된 경우에만 사용하세요.")

    selected_value = None if selected_route_name.startswith("자동") else selected_route_name
    return SidebarInputs(weights, top_n, min_passenger, growth_rate, months_ahead, user_type, time_band, selected_value, concern, use_real_llm)


def render_kpis(filtered: pd.DataFrame, future_df: pd.DataFrame) -> None:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("분석 노선 수", f"{len(filtered)}개")
    k2.metric("최고 위험 노선", f"{filtered.iloc[0]['노선명']}번" if not filtered.empty else "-")
    k3.metric("평균 CV", f"{filtered['cv_score'].mean():.3f}" if not filtered.empty else "-")
    k4.metric(
        "예측 최고 위험",
        f"{future_df['미래위험점수'].max():.1f}점" if not future_df.empty else "-",
        delta=f"{future_df['위험도변화'].max():+.1f}" if not future_df.empty else None,
    )


def render_user_alert(route: pd.Series, analysis: dict[str, str]) -> None:
    risk_class = "risk-high" if analysis["risk_label"] == "높음" else "risk-mid" if analysis["risk_label"] in ["주의", "관찰"] else "risk-low"
    st.markdown('<div class="section-title">AI 분석 결과 및 사용자 맞춤 알림</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="{risk_class}">
        <b>{route['노선명']}번 · {analysis['risk_label']} 리스크</b><br>
        {analysis['summary']}<br><br>
        <b>근거:</b> {analysis['reason']}<br>
        <b>추천 알림:</b> {analysis['recommendation']}<br>
        <b>사용자 맞춤 문구:</b> {analysis['persona_tip']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_table(top_routes: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">버스/정류장/지역 정보 표시</div>', unsafe_allow_html=True)
    show_cols = [
        "노선명",
        "운행계통",
        "등급",
        "boarding_count",
        "cv_score",
        "예상최대대기분",
        "정보접근취약점수",
        "종합위험점수",
    ]
    display = top_routes[show_cols].rename(columns={
        "노선명": "노선번호",
        "boarding_count": "평균이용객",
        "cv_score": "배차변동(CV)",
        "예상최대대기분": "예상최대대기(분)",
        "정보접근취약점수": "취약성점수",
        "종합위험점수": "종합위험점수",
    })
    st.dataframe(
        display.style.format({
            "평균이용객": "{:.0f}",
            "배차변동(CV)": "{:.3f}",
            "예상최대대기(분)": "{:.1f}",
            "취약성점수": "{:.1f}",
            "종합위험점수": "{:.1f}",
        }),
        width="stretch",
        hide_index=True,
    )


def render_map(df: pd.DataFrame, top_routes: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">노선 지도 — 위험도 상위 노선</div>', unsafe_allow_html=True)
    center = [df["latitude"].median(), df["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    for idx, row in top_routes.reset_index(drop=True).iterrows():
        route_data = df[df["노선ID"] == row["노선ID"]].sort_values("정류장순서")
        locations = list(zip(route_data["latitude"], route_data["longitude"]))
        if len(locations) < 2:
            continue
        color = COLORS[idx % len(COLORS)]
        folium.PolyLine(
            locations,
            color=color,
            weight=max(3, row["종합위험점수"] / 18),
            opacity=0.85,
            tooltip=f"{row['노선명']}번 | {row['등급']} | 위험 {row['종합위험점수']:.1f}",
        ).add_to(m)
        folium.CircleMarker(
            locations[0], radius=6, color=color, fill=True, fill_opacity=0.9, tooltip=f"기점: {row['start_station']}"
        ).add_to(m)
    st_folium(m, width=None, height=520, returned_objects=[])
    st.caption("지도는 샘플 좌표 기반입니다. 실제 서비스에서는 버스 위치/도착 API와 정류소 ID를 연결합니다.")


def render_future_prediction(future_df: pd.DataFrame, inputs: SidebarInputs) -> None:
    st.markdown('<div class="section-title">교통 취약성 또는 이용 불편 위험도 표시</div>', unsafe_allow_html=True)
    st.caption(f"예측 시나리오: 연 {inputs.growth_rate}% 수요 증가, {inputs.months_ahead}개월 후")
    if future_df.empty:
        st.warning("조건에 맞는 예측 대상 노선이 없습니다.")
        return
    pred_show = future_df[["노선명", "운행계통", "현재위험점수", "미래위험점수", "위험도변화", "등급"]]
    st.dataframe(pred_show, width="stretch", hide_index=True)
    st.bar_chart(future_df.set_index("노선명")[["현재위험점수", "미래위험점수"]], width="stretch")
    max_row = future_df.loc[future_df["미래위험점수"].idxmax()]
    st.info(
        f"조기경보: {max_row['노선명']}번은 {inputs.months_ahead}개월 후 미래 위험 {max_row['미래위험점수']:.1f}점으로 예측되어 "
        "배차 간격 점검, 대체 노선 안내, 취약 이용자 알림 강화 후보입니다."
    )


def render_model_validation(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">AI/ML 분석 근거</div>', unsafe_allow_html=True)
    train_r2, importances, n_routes, warning = train_risk_model(df)
    if warning:
        st.warning(warning)
        return
    cols = st.columns(3)
    cols[0].metric("샘플 노선 수", f"{n_routes}개")
    cols[1].metric("훈련 R²", f"{train_r2:.3f}")
    cols[2].metric("핵심 피처", max(importances, key=importances.get))
    importance_df = pd.DataFrame(importances.items(), columns=["피처", "중요도"]).sort_values("중요도", ascending=False)
    with st.expander("피처 중요도 보기"):
        st.dataframe(importance_df, width="stretch", hide_index=True)
    st.caption("MVP에서는 샘플 수가 제한되어 설명 가능한 데모 지표로 사용하며, 운영 단계에서는 더 많은 일자·시간대 데이터로 재학습해야 합니다.")


def render_ai_assistant(top_routes: pd.DataFrame, selected_route: pd.Series, inputs: SidebarInputs, analysis: dict[str, str]) -> None:
    st.markdown('<div class="section-title">심사위원용 AI 요약/정책 어시스턴트</div>', unsafe_allow_html=True)
    route_context = build_route_context(top_routes)
    default_question = (
        f"{selected_route['노선명']}번 노선을 {inputs.user_type} 관점에서 {inputs.time_band}에 이용할 때 "
        "어떤 이동 리스크와 알림 전략이 필요한지 설명해주세요."
    )
    question = st.text_area("AI에게 물어볼 질문", value=default_question, height=90)
    if st.button("AI 분석 생성", type="primary"):
        if inputs.use_real_llm:
            try:
                answer = ask_real_llm(question, route_context)
                st.success("실제 LLM 응답")
                st.markdown(answer)
            except Exception as exc:
                st.warning(f"실제 LLM 호출을 사용할 수 없어 mock 분석으로 전환했습니다: {exc}")
                st.markdown(analysis["judge_summary"])
        else:
            st.info("데모 모드: API 키 없이 규칙 기반 mock AI 분석을 사용합니다.")
            st.markdown(analysis["judge_summary"])
    with st.expander("AI가 참조하는 노선 컨텍스트"):
        st.code(route_context or "조건에 맞는 노선이 없습니다.", language="text")


def render_demo_notes() -> None:
    st.markdown('<div class="section-title">심사위원용 데모 설명</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. 왼쪽에서 사용자 유형(고령자/학생/보호자 등)과 이용 상황을 선택합니다.
        2. 노선별 수요, 배차 변동계수(CV), 정류장 수 기반 접근성 프록시가 종합 위험점수로 변환됩니다.
        3. AI 분석 영역은 위험 원인과 알림 전략을 사용자 언어로 바꿔 보여줍니다.
        4. 실제 API 키가 없어도 데모 모드로 작동하며, 운영 단계에서는 공공데이터 API와 LLM API를 연결합니다.
        """
    )
    st.markdown('<div class="section-title">한계와 향후 확장 방향</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - 현재 CSV는 샘플 데이터이므로 실시간 도착정보, 시간대별 승하차, 정류장 시설 정보가 추가되어야 합니다.
        - 실제 교통 취약성은 고령인구, 장애인 시설, 의료·학교 접근성 등 행정동 데이터를 결합해 보정해야 합니다.
        - 개인정보는 수집하지 않는 구조를 기본으로 하고, 보호자 알림 기능 도입 시 명시적 동의와 최소 수집 원칙이 필요합니다.
        - 데이터 품질 오류(누락 좌표, 중복 정류장, 노선 변경)를 탐지하는 배치 검증이 필요합니다.
        """
    )


def render_download(display_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">데모 산출물 다운로드</div>', unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    display_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button("우선순위 분석 CSV 다운로드", csv_buffer.getvalue(), "BusAlert_priority_demo.csv", "text/csv")

    report = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>BusAlert MVP 보고서</title></head>
<body><h1>BusAlert 경기도 버스 이동 리스크 AI 안내 MVP</h1>
<p>분석 노선 수: {len(display_df)}개</p>
<p>예측 대상 노선 수: {len(future_df)}개</p>
<p>공공데이터 기반 수요·배차 변동·정류장 위치를 결합해 취약 이용자 알림 우선순위를 산출합니다.</p>
</body></html>"""
    st.download_button("심사위원 데모 HTML 다운로드", report, "BusAlert_MVP_demo_report.html", "text/html")


# ─────────────────────────────────────────────
# 5. 앱 실행 흐름
# ─────────────────────────────────────────────
def main() -> None:
    configure_page()
    df = safe_load_data()
    inputs = render_sidebar(df)

    route_df = compute_route_priority(df, inputs.weights)
    filtered = route_df[route_df["boarding_count"] >= inputs.min_passenger].copy()
    if filtered.empty:
        st.warning("선택한 최소 이용객 조건에 맞는 노선이 없습니다. 사이드바 조건을 낮춰주세요.")
        st.stop()
    top_routes = filtered.head(inputs.top_n).copy()
    future_df = predict_future_risk(top_routes, inputs.growth_rate, inputs.months_ahead)
    selected_route = selected_route_or_default(filtered, inputs.selected_route_name)
    analysis = mock_ai_analysis(selected_route, inputs.user_type, inputs.time_band, inputs.concern)

    render_header()
    render_problem_and_data()
    render_kpis(filtered, future_df)
    render_user_alert(selected_route, analysis)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 현황 진단",
        "🗺️ 노선 지도",
        "🔮 미래 예측",
        "🤖 AI 분석",
        "🏁 제출 설명",
    ])
    with tab1:
        render_route_table(top_routes)
        render_model_validation(df)
        render_download(top_routes, future_df)
    with tab2:
        render_map(df, top_routes)
    with tab3:
        render_future_prediction(future_df, inputs)
    with tab4:
        render_ai_assistant(top_routes, selected_route, inputs, analysis)
    with tab5:
        render_demo_notes()


if __name__ == "__main__":
    main()
