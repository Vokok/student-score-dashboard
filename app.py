"""
Student Study Performance
Step 3: Flask 웹 대시보드
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import json
import os
import sys

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_artifacts, predict_scores

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_performance.csv")

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>학생 성적 예측 분석 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #F7F8FA;
    --white: #FFFFFF;
    --navy: #1A3558;
    --gold: #B07D2A;
    --green: #1A6640;
    --green-bg: #EAF4EE;
    --red-bg: #FDF0E4;
    --red: #854F0B;
    --text: #1E1C18;
    --mid: #5A5A5A;
    --border: rgba(0,0,0,0.09);
    --r: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Noto Sans KR', sans-serif; background: var(--bg); color: var(--text); }

  .topbar {
    background: var(--navy); color: #fff;
    padding: 16px 40px; display: flex; align-items: center; gap: 14px;
  }
  .topbar-title { font-size: 18px; font-weight: 700; }
  .topbar-sub   { font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 2px; }

  .wrap { max-width: 1160px; margin: 0 auto; padding: 32px 28px; }

  /* 탭 */
  .tabs { display: flex; gap: 4px; margin-bottom: 28px; }
  .tab {
    padding: 9px 22px; border-radius: 8px; font-size: 14px; font-weight: 500;
    cursor: pointer; border: none; background: transparent; color: var(--mid);
    transition: all .15s;
  }
  .tab.active { background: var(--navy); color: #fff; }

  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* KPI */
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 14px; margin-bottom: 28px; }
  .kpi {
    background: var(--white); border: 0.5px solid var(--border);
    border-radius: var(--r); padding: 18px 20px;
  }
  .kpi-label { font-size: 11px; color: var(--mid); font-weight: 500; margin-bottom: 6px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: var(--navy); letter-spacing: -1px; }
  .kpi-unit  { font-size: 12px; color: var(--mid); margin-left: 3px; }

  /* 차트 그리드 */
  .charts2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
  .charts3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; margin-bottom: 24px; }
  .chart-box {
    background: var(--white); border: 0.5px solid var(--border);
    border-radius: var(--r); padding: 22px;
  }
  .chart-title { font-size: 13px; font-weight: 700; margin-bottom: 16px; color: var(--text); }

  /* 예측 폼 */
  .pred-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .form-card {
    background: var(--white); border: 0.5px solid var(--border);
    border-radius: var(--r); padding: 28px;
  }
  .form-title { font-size: 15px; font-weight: 700; margin-bottom: 22px; color: var(--navy); }
  .form-group { margin-bottom: 16px; }
  .form-label { font-size: 12px; font-weight: 500; color: var(--mid); margin-bottom: 6px; display: block; }
  select {
    width: 100%; padding: 9px 12px; border: 0.5px solid var(--border);
    border-radius: 7px; font-size: 13px; font-family: inherit;
    background: var(--bg); color: var(--text); appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%235A5A5A' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 12px center;
    cursor: pointer;
  }
  .btn-predict {
    width: 100%; padding: 12px; background: var(--navy); color: #fff;
    border: none; border-radius: 8px; font-size: 14px; font-weight: 700;
    cursor: pointer; margin-top: 6px; font-family: inherit;
    transition: opacity .15s;
  }
  .btn-predict:hover { opacity: 0.88; }

  .result-card {
    background: var(--white); border: 0.5px solid var(--border);
    border-radius: var(--r); padding: 28px;
  }
  .result-title { font-size: 15px; font-weight: 700; margin-bottom: 22px; color: var(--navy); }
  .score-row { display: flex; flex-direction: column; gap: 14px; }
  .score-item {}
  .score-meta { display: flex; justify-content: space-between; margin-bottom: 6px; }
  .score-name { font-size: 13px; font-weight: 500; }
  .score-num  { font-size: 18px; font-weight: 700; color: var(--navy); }
  .bar-bg { background: #EBEBEB; border-radius: 99px; height: 8px; }
  .bar-fill { height: 8px; border-radius: 99px; transition: width .6s ease; }
  .avg-box {
    margin-top: 22px; background: var(--navy); color: #fff;
    border-radius: 10px; padding: 16px 20px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .avg-label { font-size: 13px; opacity: 0.8; }
  .avg-value { font-size: 28px; font-weight: 700; }
  .placeholder {
    display: flex; align-items: center; justify-content: center;
    height: 200px; color: var(--mid); font-size: 14px;
    border: 1.5px dashed var(--border); border-radius: 10px;
  }

  @media (max-width: 768px) {
    .charts2, .charts3, .pred-wrap { grid-template-columns: 1fr; }
    .wrap { padding: 20px 16px; }
  }
</style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="topbar-title">학생 성적 분석 · AI 예측 대시보드</div>
    <div class="topbar-sub">Study Performance Dataset · XGBoost 기반 점수 예측</div>
  </div>
</div>

<div class="wrap">

  <div class="tabs">
    <button class="tab active" onclick="showTab('overview')">데이터 개요</button>
    <button class="tab" onclick="showTab('analysis')">그룹별 분석</button>
    <button class="tab" onclick="showTab('predict')">점수 예측</button>
  </div>

  <!-- ===== 탭1: 개요 ===== -->
  <div id="tab-overview" class="tab-pane active">
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">총 학생 수</div><div class="kpi-value"><span id="kpi-total">-</span><span class="kpi-unit">명</span></div></div>
      <div class="kpi"><div class="kpi-label">수학 평균</div><div class="kpi-value"><span id="kpi-math">-</span><span class="kpi-unit">점</span></div></div>
      <div class="kpi"><div class="kpi-label">읽기 평균</div><div class="kpi-value"><span id="kpi-read">-</span><span class="kpi-unit">점</span></div></div>
      <div class="kpi"><div class="kpi-label">쓰기 평균</div><div class="kpi-value"><span id="kpi-write">-</span><span class="kpi-unit">점</span></div></div>
      <div class="kpi"><div class="kpi-label">시험 준비 완료율</div><div class="kpi-value"><span id="kpi-prep">-</span><span class="kpi-unit">%</span></div></div>
    </div>
    <div class="charts2">
      <div class="chart-box"><div class="chart-title">수학 점수 분포</div><canvas id="histMath" height="180"></canvas></div>
      <div class="chart-box"><div class="chart-title">읽기 / 쓰기 점수 분포</div><canvas id="histRW" height="180"></canvas></div>
    </div>
    <div class="charts2">
      <div class="chart-box"><div class="chart-title">성별 평균 점수</div><canvas id="barGender" height="180"></canvas></div>
      <div class="chart-box"><div class="chart-title">시험 준비 여부별 평균 점수</div><canvas id="barPrep" height="180"></canvas></div>
    </div>
  </div>

  <!-- ===== 탭2: 그룹별 분석 ===== -->
  <div id="tab-analysis" class="tab-pane">
    <div class="charts2">
      <div class="chart-box"><div class="chart-title">인종 그룹별 수학 평균</div><canvas id="barRace" height="200"></canvas></div>
      <div class="chart-box"><div class="chart-title">부모 학력별 평균 점수</div><canvas id="barParent" height="200"></canvas></div>
    </div>
    <div class="charts2">
      <div class="chart-box"><div class="chart-title">점심 유형별 평균 점수</div><canvas id="barLunch" height="200"></canvas></div>
      <div class="chart-box"><div class="chart-title">모델 성능 (R² / MAE)</div><canvas id="barModel" height="200"></canvas></div>
    </div>
  </div>

  <!-- ===== 탭3: 점수 예측 ===== -->
  <div id="tab-predict" class="tab-pane">
    <div class="pred-wrap">
      <div class="form-card">
        <div class="form-title">학생 조건 입력</div>
        <div class="form-group">
          <label class="form-label">성별</label>
          <select id="sel-gender">
            <option value="female">여성 (female)</option>
            <option value="male">남성 (male)</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">인종 그룹</label>
          <select id="sel-race">
            <option value="group A">group A</option>
            <option value="group B">group B</option>
            <option value="group C" selected>group C</option>
            <option value="group D">group D</option>
            <option value="group E">group E</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">부모 최종 학력</label>
          <select id="sel-edu">
            <option value="some high school">고등학교 일부 (some high school)</option>
            <option value="high school">고등학교 졸업 (high school)</option>
            <option value="some college">대학교 일부 (some college)</option>
            <option value="associate's degree">전문대 졸업 (associate's degree)</option>
            <option value="bachelor's degree" selected>학사 (bachelor's degree)</option>
            <option value="master's degree">석사 (master's degree)</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">점심 유형</label>
          <select id="sel-lunch">
            <option value="standard" selected>일반 (standard)</option>
            <option value="free/reduced">무상/할인 (free/reduced)</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">시험 준비 완료 여부</label>
          <select id="sel-prep">
            <option value="none" selected>미완료 (none)</option>
            <option value="completed">완료 (completed)</option>
          </select>
        </div>
        <button class="btn-predict" onclick="runPredict()">점수 예측하기</button>
      </div>

      <div class="result-card">
        <div class="result-title">예측 결과</div>
        <div id="pred-result">
          <div class="placeholder">조건을 입력 후 예측 버튼을 눌러주세요</div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
let globalData = null;

function showTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

function makeHist(ctx, data, label, color) {
  const bins = 20;
  const mn = Math.min(...data), mx = Math.max(...data);
  const step = (mx - mn) / bins;
  const counts = Array(bins).fill(0);
  const labels = [];
  for (let i = 0; i < bins; i++) labels.push(Math.round(mn + step * i));
  data.forEach(v => {
    let idx = Math.floor((v - mn) / step);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  });
  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data: counts, backgroundColor: color, borderRadius: 3 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: '점수' } }, y: { title: { display: true, text: '학생 수' } } } }
  });
}

function makeGroupBar(ctx, groups, datasets, title) {
  const colors = ['rgba(26,53,88,0.8)', 'rgba(176,125,42,0.8)', 'rgba(26,102,64,0.8)'];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: groups,
      datasets: datasets.map((d, i) => ({ label: d.label, data: d.data, backgroundColor: colors[i], borderRadius: 3 }))
    },
    options: { plugins: { legend: { position: 'top' } }, scales: { y: { min: 40, title: { display: true, text: '평균 점수' } } } }
  });
}

async function loadData() {
  const res = await fetch('/api/overview');
  globalData = await res.json();
  const d = globalData;

  // KPI
  document.getElementById('kpi-total').textContent = d.total.toLocaleString();
  document.getElementById('kpi-math').textContent  = d.avg_math;
  document.getElementById('kpi-read').textContent  = d.avg_reading;
  document.getElementById('kpi-write').textContent = d.avg_writing;
  document.getElementById('kpi-prep').textContent  = d.prep_rate;

  // 히스토그램
  makeHist(document.getElementById('histMath'), d.math_scores, '수학', 'rgba(26,53,88,0.75)');

  new Chart(document.getElementById('histRW'), {
    type: 'bar',
    data: {
      labels: d.score_bins,
      datasets: [
        { label: '읽기', data: d.reading_hist, backgroundColor: 'rgba(176,125,42,0.7)', borderRadius: 3 },
        { label: '쓰기', data: d.writing_hist, backgroundColor: 'rgba(26,102,64,0.7)', borderRadius: 3 }
      ]
    },
    options: { plugins: { legend: { position: 'top' } }, scales: { x: { title: { display: true, text: '점수' } }, y: { title: { display: true, text: '학생 수' } } } }
  });

  // 성별
  makeGroupBar(document.getElementById('barGender'), d.gender_groups,
    [{ label: '수학', data: d.gender_math }, { label: '읽기', data: d.gender_reading }, { label: '쓰기', data: d.gender_writing }]);

  // 시험준비
  makeGroupBar(document.getElementById('barPrep'), d.prep_groups,
    [{ label: '수학', data: d.prep_math }, { label: '읽기', data: d.prep_reading }, { label: '쓰기', data: d.prep_writing }]);

  // 탭2
  new Chart(document.getElementById('barRace'), {
    type: 'bar',
    data: {
      labels: d.race_groups,
      datasets: [
        { label: '수학', data: d.race_math, backgroundColor: 'rgba(26,53,88,0.8)', borderRadius: 3 },
        { label: '읽기', data: d.race_reading, backgroundColor: 'rgba(176,125,42,0.8)', borderRadius: 3 },
      ]
    },
    options: { plugins: { legend: { position: 'top' } }, scales: { y: { min: 50 } } }
  });

  new Chart(document.getElementById('barParent'), {
    type: 'bar',
    data: {
      labels: d.parent_groups,
      datasets: [{ label: '수학 평균', data: d.parent_math, backgroundColor: 'rgba(26,53,88,0.75)', borderRadius: 3 }]
    },
    options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { min: 50 } } }
  });

  new Chart(document.getElementById('barLunch'), {
    type: 'bar',
    data: {
      labels: d.lunch_groups,
      datasets: [
        { label: '수학', data: d.lunch_math, backgroundColor: 'rgba(26,53,88,0.8)', borderRadius: 3 },
        { label: '읽기', data: d.lunch_reading, backgroundColor: 'rgba(176,125,42,0.8)', borderRadius: 3 },
        { label: '쓰기', data: d.lunch_writing, backgroundColor: 'rgba(26,102,64,0.8)', borderRadius: 3 },
      ]
    },
    options: { plugins: { legend: { position: 'top' } }, scales: { y: { min: 50 } } }
  });

  // 모델 성능
  new Chart(document.getElementById('barModel'), {
    type: 'bar',
    data: {
      labels: ['수학', '읽기', '쓰기'],
      datasets: [
        { label: 'R²', data: d.model_r2, backgroundColor: 'rgba(26,53,88,0.8)', borderRadius: 3, yAxisID: 'y' },
        { label: 'MAE (점)', data: d.model_mae, backgroundColor: 'rgba(176,125,42,0.8)', borderRadius: 3, yAxisID: 'y2' },
      ]
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: {
        y:  { position: 'left',  title: { display: true, text: 'R²' }, min: 0, max: 1 },
        y2: { position: 'right', title: { display: true, text: 'MAE (점)' }, grid: { drawOnChartArea: false } }
      }
    }
  });
}

async function runPredict() {
  const payload = {
    gender: document.getElementById('sel-gender').value,
    race_ethnicity: document.getElementById('sel-race').value,
    parental_level_of_education: document.getElementById('sel-edu').value,
    lunch: document.getElementById('sel-lunch').value,
    test_preparation_course: document.getElementById('sel-prep').value,
  };
  const res  = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();

  const colors = { math_score: '#1A3558', reading_score: '#B07D2A', writing_score: '#1A6640' };
  const labels = { math_score: '수학', reading_score: '읽기', writing_score: '쓰기' };
  const targets = ['math_score', 'reading_score', 'writing_score'];

  let html = '<div class="score-row">';
  targets.forEach(t => {
    const val = data[t];
    html += `<div class="score-item">
      <div class="score-meta">
        <span class="score-name">${labels[t]}</span>
        <span class="score-num">${val}<span style="font-size:13px;font-weight:400;color:var(--mid)">점</span></span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:${val}%;background:${colors[t]}"></div></div>
    </div>`;
  });
  html += '</div>';
  html += `<div class="avg-box"><span class="avg-label">예측 평균 점수</span><span class="avg-value">${data.average}점</span></div>`;
  document.getElementById('pred-result').innerHTML = html;
}

loadData();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/api/overview")
def api_overview():
    df = pd.read_csv(DATA_PATH)
    _, encoders, meta, eda = load_artifacts()

    def hist_bins(series, bins=20):
        mn, mx = series.min(), series.max()
        step = (mx - mn) / bins
        counts = [0] * bins
        labels = [round(mn + step * i) for i in range(bins)]
        for v in series:
            idx = min(int((v - mn) / step), bins - 1)
            counts[idx] += 1
        return labels, counts

    math_labels, _ = hist_bins(df["math_score"])
    _, read_hist   = hist_bins(df["reading_score"])
    _, write_hist  = hist_bins(df["writing_score"])

    def grp_avgs(col, targets):
        g = df.groupby(col)[targets].mean().round(1)
        return g.index.tolist(), [g[t].tolist() for t in targets]

    g_groups, (g_math, g_read, g_write) = grp_avgs("gender", ["math_score","reading_score","writing_score"])
    p_groups, (p_math, p_read, p_write) = grp_avgs("test_preparation_course", ["math_score","reading_score","writing_score"])
    r_groups, (r_math, r_read, _)        = grp_avgs("race_ethnicity", ["math_score","reading_score","writing_score"])
    pa_groups, (pa_math, _, _)           = grp_avgs("parental_level_of_education", ["math_score","reading_score","writing_score"])
    l_groups, (l_math, l_read, l_write)  = grp_avgs("lunch", ["math_score","reading_score","writing_score"])

    metrics = meta["metrics"]

    payload = {
        "total":       len(df),
        "avg_math":    round(df["math_score"].mean(), 1),
        "avg_reading": round(df["reading_score"].mean(), 1),
        "avg_writing": round(df["writing_score"].mean(), 1),
        "prep_rate":   round((df["test_preparation_course"] == "completed").mean() * 100, 1),
        "math_scores": df["math_score"].tolist(),
        "score_bins":  math_labels,
        "reading_hist": read_hist,
        "writing_hist": write_hist,
        "gender_groups":  g_groups, "gender_math": g_math, "gender_reading": g_read, "gender_writing": g_write,
        "prep_groups":    p_groups, "prep_math":   p_math, "prep_reading":   p_read, "prep_writing":   p_write,
        "race_groups":    r_groups, "race_math":   r_math, "race_reading":   r_read,
        "parent_groups":  pa_groups,"parent_math": pa_math,
        "lunch_groups":   l_groups, "lunch_math":  l_math, "lunch_reading":  l_read, "lunch_writing":  l_write,
        "model_r2":  [metrics["math_score"]["r2"], metrics["reading_score"]["r2"], metrics["writing_score"]["r2"]],
        "model_mae": [metrics["math_score"]["mae"],metrics["reading_score"]["mae"],metrics["writing_score"]["mae"]],
    }
    return app.response_class(
        json.dumps(payload, cls=NumpyEncoder, ensure_ascii=False),
        mimetype="application/json"
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    from flask import request as req
    data = req.get_json()
    result = predict_scores(data)
    return app.response_class(
        json.dumps(result, cls=NumpyEncoder, ensure_ascii=False),
        mimetype="application/json"
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  학생 성적 예측 대시보드")
    print("  http://127.0.0.1:5001")
    print("=" * 50)
    app.run(debug=False, port=5001)
