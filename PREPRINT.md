# PREPRINT.md — скелет препринта (вектор 2, 2026-11-23)

Цель: препринт для вычислительных физиков. Целевая площадка: **arXiv
physics.comp-ph** (предварительный; позже — воркшоп по решётным/численным
методам). GitHub-публикация и CI — **завершены 2026-11-24**: репо
**https://github.com/HouKonaizer/wm-model5** (публичный, MIT), CI — GitHub Actions
(Python 3.14, numpy==2.5.2, pytest 131 + `wm checks --fast`), зелёный.

**Рабочее название:**
*"A Self-Consistent Five-Subspace Lattice Model with an Honest
Calibration/Prediction Audit"*

## Abstract (EN, ~190 слов)

We present WM, a self-consistent lattice model in which the world is a
five-subspace volumetric torus (time, energy, probability, gravity, fields)
evolved by a single split-step kernel. The core conserves node count,
momentum, total energy and charge to ~1e-15, and its numerics are
bit-locked (κ = 0.4069113352735587) for reproducibility. A scale map with
four anchors (hadronization energy U, fine structure constant α, h, dt) and
five free dials maps model units to SI. The central contribution is an
*honest audit* of 83 quantitative checks against experiment: 4 calibrations
(identity by construction), 50 standard formulas (deviation ≈ 0,
implementation checks), 12 predictions outside calibration (e.g. a_e at
+7×10⁻⁶ % with the 3-loop QED term, R_π at −0.18 %, R_K at +0.65 % against
KLOE, parapositronium lifetime at +0.65 % — a documented tension, the 21 cm
line at +0.053 % with the leading-order tree), and 17 honest lattice
computations separating artifacts from physics. We explicitly do *not* claim
a derivation of α, G or m_e from first principles: they are inputs, as is
the particle spectrum. The gravity sector is a documented closed boundary
(model G = 1.6×10³⁷ × physical G); the GPS redshift coefficient
κ = 0.407 (−59.3 % vs GR) and the falsified prediction P1 (β_g = 1) are
recorded as boundaries, not refutations. Five frozen predictions (P1–P5)
carry explicit falsification protocols. All numbers are machine-readable
(verification.json, offline dashboard audit.html) and reproducible (131 tests, `wm checks`).

## Структура препринта (план разделов)

1. **Introduction** — постановка: численная платформа, а не теория новой
   физики; отличие от «toy-моделей» — честная классификация каждого числа
   (калитка: calibration / standard / prediction / lattice).
2. **Model** — 5 субпространств L0–L4; вакуум/материя; энтропийный регулятор;
   единое ядро SSFM (V = 2g + E, унитарность L2); SI-карта (4 якоря + 5
   диалов; модельные единицы m_e, ħ, g_e, a₀, E₁).
3. **Numerical core** — сохранения N, p, ΣE, Σq ~1e-15; бит-лок κ D12;
   сходимости D7–D11; разделение артефакт/физика (D1–D6, D15a/b, D17);
   воспроизводимость (сид, детерминизм).
4. **Verification audit** — таблица 83 сверок (4/50/12/17); методология
   классификации; `verification.json` как машинный вывод;
    `audit.html` — офлайн-дашборд аудита (4 таба: сверки / предсказания / P1–P5 / о модели).
5. **Predictions and falsifiability** — P1–P5 (значение, погрешность, внешний
   способ, критерий 3σ); P1 фальсифицирован при β_g = 1 — как граница;
   QED-блок: a_e 3-loop, R_π/R_K (SM+RC, arXiv-источники), p-Ps (напряжение
   SM>exp), 21 см (LO; полный tree — HOLD).
6. **Limitations and boundaries** — G1 (G-сектор, 1.57e37), κ_GPS −59.3%,
   P5 (−99.76%), свободные диалы (G4), нумерология; что НЕ заявляется.
7. **Reproducibility** — `pip install -e .`, `wm demo`, `wm run`, `wm
   checks`, `pytest tests` (131); бит-лок; CI — GitHub Actions (см. Data &
   Code Availability).
8. **Conclusions** — ценность: единое ядро с проверяемыми сохранениями +
   честный аудит + замороженные фальсифицируемые предсказания; путь
   вперёд: вывод хотя бы одного диала/якоря из динамики ядра.

## Claims / Non-claims (ключевая секция для рецензента)

**Заявляем:**
- самосогласованное юнитарное ядро с сохранностями до ~1e-15 и бит-локом;
- полную классификацию 83 сверок (каждое число помечено классом);
- QED-совпадения в пределах ~0.1–17% — как *реализация SM-формул* через
  SI-карту модели;
- задокументированные границы (kill G1–G5, P1 falsified, P5, 21 см HOLD).

**Не заявляем:**
- вывод α, G, m_e из первых принципов (входы);
- новую физику (совпадения = SM-формулы);
- объяснение κ_GPS (это расхождение, зафиксированное как граница);
- сходство p-Ps (SM выше экспа на +0.65% — напряжение).

## Рисунки (предложение)

1. Срез 3 слоёв (L1/L2/L3) + узлы — из viewer.html (кадр).
2. Таблица 83 сверок (выжимка) с цветовкой по классу (или скрин `audit.html`).
3. Таблица 12 предсказаний: dev в %, с σ-оценками там, где есть.
4. Бит-лок: κ(t) при t = 10…50 (0.3887 → 0.4042) — «нет уникального слабого
   предела».

## Data & Code Availability

- `verification.json` — машинный вывод полного аудита (83).
- `audit.html` — офлайн-дашборд того же аудита (генерируется `wm checks`).
- 131 pytest-тестов; `wm checks` пересобирает аудит за ~3 мин.
- GitHub-репозиторий (публичный, MIT, CI — GitHub Actions, зелёный):
  https://github.com/HouKonaizer/wm-model5

## Язык препринта

Английский (abstract и разделы — EN; цитируемые RU-доки — как
supplementary). Текущий документ — скелет; текст разделов писать по мере
свежести состояния проекта.
