"""
따릉이 기후데이터 분석
기온/미세먼지와 서울시 공공자전거 이용 패턴 변화 분석
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── 한글 폰트 설정 ────────────────────────────────────────────
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# Step 1: 데이터 로딩
# ============================================================
print("=" * 50)
print("Step 1: 데이터 로딩")
print("=" * 50)

df_bike = pd.read_csv('ddareungi.csv', encoding='utf-8-sig')
df_weather = pd.read_csv('weather.csv', encoding='utf-8-sig')

print("따릉이 데이터 shape:", df_bike.shape)
print(df_bike.head())
print("\n기상 데이터 shape:", df_weather.shape)
print(df_weather.head())

# ============================================================
# Step 2: 데이터 전처리
# ============================================================
print("\n" + "=" * 50)
print("Step 2: 데이터 전처리")
print("=" * 50)

# 1) 날짜 컬럼 통일
df_bike['date'] = pd.to_datetime(df_bike['대여일시']).dt.strftime('%Y-%m-%d')
df_weather['date'] = pd.to_datetime(df_weather['날짜']).dt.strftime('%Y-%m-%d')

# 2) 따릉이: 일별 대여건수 집계
df_bike_daily = df_bike.groupby('date').size().reset_index(name='이용건수')

# 3) 두 데이터 날짜 기준 병합
df = pd.merge(df_bike_daily, df_weather, on='date', how='inner')

# 4) 결측값 처리
print("결측값 현황:\n", df.isnull().sum())
df = df.dropna(subset=['평균기온', 'PM10', '이용건수'])

# 5) 파생변수 생성
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.day_name()
df['season'] = df['month'].map({
    12: '겨울', 1: '겨울', 2: '겨울',
    3: '봄',  4: '봄',  5: '봄',
    6: '여름', 7: '여름', 8: '여름',
    9: '가을', 10: '가을', 11: '가을',
})

# 6) 미세먼지 PM10 등급 범주화 (환경부 기준)
def pm_grade(x):
    if x <= 30:   return '좋음'
    elif x <= 80: return '보통'
    elif x <= 150: return '나쁨'
    else:          return '매우나쁨'

df['미세먼지등급'] = df['PM10'].apply(pm_grade)
grade_order = ['좋음', '보통', '나쁨', '매우나쁨']
df['미세먼지등급'] = pd.Categorical(df['미세먼지등급'], categories=grade_order, ordered=True)

# 7) 기온 구간 생성 (5°C 단위)
bins   = [-20, 0, 5, 10, 15, 20, 25, 30, 40]
labels = ['-20~0°C', '0~5°C', '5~10°C', '10~15°C', '15~20°C', '20~25°C', '25~30°C', '30°C+']
df['기온구간'] = pd.cut(df['평균기온'], bins=bins, labels=labels, right=False)

print("\n전처리 완료. 최종 shape:", df.shape)
print(df[['date', '이용건수', '평균기온', 'PM10', 'season', '미세먼지등급']].head(10))

# ============================================================
# Step 3: 수치형 분석 (기온 ↔ 이용량)
# ============================================================
print("\n" + "=" * 50)
print("수치형 분석: 기온과 이용량의 상관관계")
print("=" * 50)

# 피어슨 상관계수 — 기온
r, p = stats.pearsonr(df['평균기온'], df['이용건수'])
print(f"\n[기온-이용량 피어슨 상관계수] r = {r:.4f}, p-value = {p:.4e}")
if p < 0.05:
    print("→ 통계적으로 유의미한 상관관계 (p < 0.05)")

# 기온 구간별 평균 이용량
temp_group = df.groupby('기온구간', observed=True)['이용건수'].agg(['mean', 'std', 'count'])
temp_group.columns = ['평균이용건수', '표준편차', '데이터수']
print("\n[기온 구간별 평균 이용건수]")
print(temp_group.round(0))

# 피어슨 상관계수 — PM10
r2, p2 = stats.pearsonr(df['PM10'], df['이용건수'])
print(f"\n[PM10-이용량 피어슨 상관계수] r = {r2:.4f}, p-value = {p2:.4e}")

print("\n[이용건수 기술통계]")
print(df['이용건수'].describe().round(0))

# ============================================================
# Step 4: 범주형 분석 (미세먼지 등급, 계절, 요일)
# ============================================================
print("\n" + "=" * 50)
print("범주형 분석")
print("=" * 50)

# 미세먼지 등급별 평균 이용량
pm_group = df.groupby('미세먼지등급', observed=True)['이용건수'].mean().round(0)
print("\n[미세먼지 등급별 평균 이용건수]")
print(pm_group)

base = pm_group['좋음']
for grade in ['보통', '나쁨', '매우나쁨']:
    if grade in pm_group.index:
        ratio = (pm_group[grade] - base) / base * 100
        print(f"  '{grade}' 등급은 '좋음' 대비 {ratio:+.1f}%")

# 계절별 평균 이용량
season_order = ['봄', '여름', '가을', '겨울']
season_group = df.groupby('season')['이용건수'].mean().reindex(season_order).round(0)
print("\n[계절별 평균 이용건수]")
print(season_group)

# 요일별 평균 이용량
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_kr = {'Monday': '월', 'Tuesday': '화', 'Wednesday': '수',
              'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'}
weekday_group = df.groupby('weekday')['이용건수'].mean().reindex(weekday_order).round(0)
weekday_group.index = [weekday_kr[d] for d in weekday_group.index]
print("\n[요일별 평균 이용건수]")
print(weekday_group)

# ============================================================
# Step 5: 시각화 (figure1.png ~ figure4.png)
# ============================================================
print("\n" + "=" * 50)
print("Step 5: 시각화")
print("=" * 50)

colors_main   = '#2D7A5E'
colors_accent = '#F4A942'

# ── 그래프 1: 기온과 이용량 산점도 + 회귀선 ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['평균기온'], df['이용건수'], alpha=0.3, color=colors_main, s=15, label='일별 데이터')

m, b_reg = np.polyfit(df['평균기온'], df['이용건수'], 1)
x_line = np.linspace(df['평균기온'].min(), df['평균기온'].max(), 100)
ax.plot(x_line, m * x_line + b_reg, color=colors_accent, linewidth=2.5,
        label=f'회귀선 (r={r:.2f})')

ax.set_xlabel('평균기온 (°C)', fontsize=12)
ax.set_ylabel('일별 이용건수', fontsize=12)
ax.set_title('기온과 따릉이 이용량의 상관관계', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1.png', dpi=150)
plt.close()
print("figure1.png 저장 완료")

# ── 그래프 2: 미세먼지 등급별 박스플롯 ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
palette = {'좋음': '#2D7A5E', '보통': '#4CAF82', '나쁨': '#F4A942', '매우나쁨': '#E53935'}
sns.boxplot(data=df, x='미세먼지등급', y='이용건수', hue='미세먼지등급',
            order=grade_order, palette=palette, ax=ax, width=0.55, legend=False)

ax.set_xlabel('미세먼지 등급 (PM10)', fontsize=12)
ax.set_ylabel('일별 이용건수', fontsize=12)
ax.set_title('미세먼지 등급별 따릉이 이용량 분포', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure2.png', dpi=150)
plt.close()
print("figure2.png 저장 완료")

# ── 그래프 3: 월별 평균 이용량 라인차트 ──────────────────────
monthly = df.groupby('month')['이용건수'].mean().round(0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(monthly.index, monthly.values, marker='o', color=colors_main,
        linewidth=2.5, markersize=8)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=colors_main)

for x, y in zip(monthly.index, monthly.values):
    ax.annotate(f'{int(y):,}', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9)

ax.set_xticks(range(1, 13))
ax.set_xticklabels([f'{i}월' for i in range(1, 13)])
ax.set_xlabel('월', fontsize=12)
ax.set_ylabel('평균 이용건수', fontsize=12)
ax.set_title('월별 따릉이 평균 이용량 변화', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3.png', dpi=150)
plt.close()
print("figure3.png 저장 완료")

# ── 그래프 4: 계절별 막대그래프 ─────────────────────────────
season_colors = {'봄': '#4CAF82', '여름': '#F4A942', '가을': '#2D7A5E', '겨울': '#1A3C34'}
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(season_group.index, season_group.values,
              color=[season_colors[s] for s in season_group.index],
              width=0.55, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, season_group.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{int(val):,}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('계절', fontsize=12)
ax.set_ylabel('평균 이용건수', fontsize=12)
ax.set_title('계절별 따릉이 평균 이용량', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure4.png', dpi=150)
plt.close()
print("figure4.png 저장 완료")

print("\n✅ 모든 분석 및 시각화 완료!")
print("저장된 파일: figure1.png ~ figure4.png")
