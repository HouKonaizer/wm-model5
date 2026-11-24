# -*- coding: utf-8 -*-
"""
ЦИФРОВАЯ МОДЕЛЬ 5 СУБПРОСТРАНСТВ В 3D ОБЪЁМЕ
============================================
Субпространства:
  L0: ВРЕМЯ       (направленность, дилатация, причинность)
  L1: ЭНЕРГИЯ     (консервация, квантование, нулевая энергия)
  L2: ВЕРОЯТНОСТЬ (суперпозиция, нелокальность, интерференция)
  L3: ГРАВИТАЦИЯ  (кривизна, притяжение, замедление времени)
  L4: ПОЛЯ        (электромагнетизм, дипольность, калибровка)

Энтропийный регулятор:
  Мониторит глобальную энтропию и применяет давление
  для удержания системы в динамическом равновесии.

Теория:
  - Мир = объединение субпространств (парадигм)
  - Вакуум = раслоение; материя = узел (синтез), где слои «в фазе»
  - Связности — базовые элементы; явления = n-комбинации связностей
  - Тепло = частичное раслоение (пример: ломаем дерево)

Свойства подобраны по экспериментам:
  Hafele-Keating (1971), мюоны, GPS (+38.5 мкс/день: грав. +45.7, кинемат. −7.2), LIGO (2015),
  Белл/Aspect (1982), двойная щель (Tonomura 1989), Штерн-Герлах (1922),
  аннигиляция e+e- -> 2 gamma (E=mc^2), эффект Казимира.

Запуск:  python model.py (совместимый shim в корне) или: wm run / python -m wm
Результат: терминальный вывод + snapshots.json + viewer.html (3D, открыть в браузере)
"""

import json
import math
import os
import unicodedata

import numpy as np

# ============================================================
# КОНСТАНТЫ И ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ============================================================

# Размер 3D сетки
GRID_SIZE = 20  # 20x20x20 = 8000 ячеек

# Индексы субпространств
TIME, ENERGY, PROBABILITY, GRAVITY, FIELDS = 0, 1, 2, 3, 4
N_LAYERS = 5
LAYER_NAMES = ["ВРЕМЯ", "ЭНЕРГИЯ", "ВЕРОЯТНОСТЬ", "ГРАВИТАЦИЯ", "ПОЛЯ"]

# --- L0: ВРЕМЯ (Hafele-Keating 1971: 59 нс на часах; мюоны; GPS) ---
TIME_FLOW = 1.0              # Направление: всегда вперёд (стрелка времени)
TIME_DILATION_GRAV = 0.3     # Коэфф. гравитационного замедления (ОТО)
TIME_DILATION_VEL = 0.1      # Коэфф. релятивистского замедления (СТО)

# --- L1: ЭНЕРГИЯ (аннигиляция e+e- -> 2 gamma; синтез; Казимир) ---
# Консервация ΣE — не флаг, а тождество схемы (попарно-антисимметричные
# потоки; ledger и сверка D9: ΔΣE = ledger до машинной точности).
ENERGY_ZERO_POINT = 0.02     # Нулевая энергия вакуума
ENERGY_FLOW_RATE = 0.05      # Скорость перетекания (аналог теплопроводности)
GRAV_ENERGY_COUPLING = 0.10  # Энергия «падает» в грав. ямы (консервативно, ΣE=const)

# --- L2: ВЕРОЯТНОСТЬ (двойная щель; Белл/Aspect 1982; Гейзенберг) ---
PROB_INTERFERENCE = 0.8      # Сила интерференции (двойной щель)
PROB_NONLOCALITY = 0.3       # Нелокальность (запутанность, Белл)
PROB_UNCERTAINTY = 0.1       # Неопределённость (Гейзенберг)

# --- L3: ГРАВИТАЦИЯ (LIGO 2015; GPS +38.5 мкс/день: грав. +45.7, кинемат. −7.2) ---
GRAV_STRENGTH = 0.08         # Сила (слабая: 10^-36 от ЭМ в масштабе модели)
GRAV_SPEED = 1.0             # Скорость распространения = c (LIGO)

# --- L4: ПОЛЯ (Штерн-Герлах 1922; Холл 1980; Фарадей) ---
FIELD_DIPOLE = True          # Дипольность (+/-)
FIELD_STRENGTH = 0.4         # Сила (сильная на человеческом масштабе)
FIELD_GAUGE = 0.1            # Калибровочная симметрия U(1)

# --- МАТЕРИЯ (узел/синтез) ---
NODE_BIND_STRENGTH = 0.9    # Связность узла: поток энергии через узел (1-B)*c0
PSI_NODE_AMP = 0.7          # Амплитуда psi в узле (связанное состояние)
MATTER_LIFETIME = 250       # Срок жизни узла (аналог периодического распада)
MATTER_REST_MASS = 0.6      # Собственная масса узла (E=mc^2): гравитирует всегда

# --- Типы узлов (каталог частиц, particles.py) ---
from .particles import BY_SYMBOL as _CATALOG, HADRON_COMPOSITION

_ANTI_U = "\u016b"  # анти-up кварк (ū, как в каталоге particles.ANTI)
_ANTI_D = "d\u0304"  # анти-down кварк (d̄, как в каталоге particles.ANTI)
_NODE_TYPES = ["n", "p", "e⁻", "μ⁻", "τ⁻", "π±", "γ", "u", _ANTI_U, "d", _ANTI_D]
TYPE_CODE = {s: i for i, s in enumerate(_NODE_TYPES)}

# Палитра типов узлов для анимации (индекс = код типа): [r, g, b]
NODE_COLORS = {
    0: [0.62, 0.67, 0.75],    # n
    1: [1.00, 0.85, 0.35],    # p
    2: [0.30, 0.85, 1.00],    # e⁻
    3: [0.85, 0.35, 0.95],    # μ⁻
    4: [1.00, 0.45, 0.70],    # τ⁻
    5: [1.00, 0.55, 0.25],    # π±
    6: [1.00, 1.00, 1.00],    # γ
    7: [1.00, 0.35, 0.35],    # u
    8: [0.72, 0.20, 0.20],    # ū
    9: [0.35, 0.48, 1.00],    # d
    10: [0.62, 0.83, 1.00],   # d̄
}


def _base_symbol(sym: str) -> str:
    """'ū' → 'u': античастица = та же масса/спин (заряд в решётной
    динамике не отслеживается)."""
    nfd = unicodedata.normalize("NFD", sym)
    return nfd.replace("\u0304", "")


def _particle(sym: str):
    return _CATALOG.get(_base_symbol(sym))


def _node_charge(sym: str) -> float:
    """Заряд узла (в единицах |e|); античастица (макрон) — со знаком минус."""
    p = _particle(sym)
    if p is None:
        return 0.0
    q = p.charge
    if sym != p.symbol:   # анти-частица: инверсия заряда
        q = -q
    return q


def _node_meff(sym: str) -> float:
    """Масса узла в единицах массы протона (m/m_p); m = 0 → 0 (не двигается)."""
    p = _particle(sym)
    if p is None or p.mass_eV <= 0:
        return 0.0
    return p.mass_eV / _CATALOG["p"].mass_eV


_NODE_CHARGE = {s: _node_charge(s) for s in _NODE_TYPES}
_NODE_MEFF = {s: _node_meff(s) for s in _NODE_TYPES}


# --- Lifetime: ЛОГАРИФМИЧЕСКОЕ СЖАТИЕ ---
# Физические τ разбросаны на 15 порядков (2.9e-13 с [τ] — 878 с [n]);
# линейная шкала не позволяет визуализировать цепочки распадов.
# Сжатие: τ_min → 50 шагов, τ_max → 250 шагов (порядок сохраняется).
_dyn_t = [_CATALOG[_base_symbol(s)].lifetime_s
          for s in _NODE_TYPES
          if _CATALOG.get(_base_symbol(s)) is not None
          and _CATALOG.get(_base_symbol(s)).lifetime_s is not None]
_T_MIN, _T_MAX = min(_dyn_t), max(_dyn_t)


def _dyn_lifetime(sym: str):
    p = _particle(sym)
    if p is None or p.lifetime_s is None:
        return None
    t = p.lifetime_s
    if t <= _T_MIN:
        return 50
    if t >= _T_MAX:
        return MATTER_LIFETIME
    f = (math.log10(t) - math.log10(_T_MIN)) / (math.log10(_T_MAX) - math.log10(_T_MIN))
    return int(50 + (MATTER_LIFETIME - 50) * f)


TYPE_LIFETIME = {s: _dyn_lifetime(s) for s in _NODE_TYPES}

# Каналы распада (демо; качественно по BR из PDG):
#   τ⁻: e⁻ (17.8%), μ⁻ (17.4%), π± (адронизация, 64.8%)
#   μ⁻: e⁻ (100%);  π±: μ⁻ (99.99%)
TYPE_DECAY = {
    "τ⁻": [("e⁻", 0.178), ("μ⁻", 0.174), ("π±", 0.648)],
    "μ⁻": [("e⁻", 1.0)],
    "π±": [("μ⁻", 1.0)],
}

# Статистика: спин 1/2 в каталоге = фермион (Pauli), иначе бозон (стек)
FERMI_TYPES = {s for s in _NODE_TYPES
               if _particle(s) is not None and _particle(s).spin == 0.5}

# --- Струна (конфайнмент): V = σ·d; d ≥ STRING_BREAK → разрыв + пара q q̄ ---
STRING_BREAK = 4
STRING_TENSION = 0.05
# Бозе: число ДОП. бозонов сверх 2 слотов на ячейку (макс. 3 узла на ячейку)
BOSE_STACK_MAX = 1

# --- ЭМ-ДИНАМИКА (фазы 10–11): кулоновское взаимодействие заряженных узлов ---
# COULOMB_K = 3.6e-4 — α-якорь (K·U·h = α·ħ·c, S1, THEORY.md §4, F4) и при нём
# круговая кулонова орбита электрона вокруг протона (r0 = 3 ячейки) стабильна
# за 250+ шагов: симплициев Евлер + консервативное сглаживание Платтера
# (энергия ограничена, орбита не затухает). Квантовый H (D19/D20): в модельных
# единицах это даёт a₀ = 10344.26 яч., E₁ = 13.6057 эВ (α-якорь ⇒ соответствие
# по построению, G2); классическая орбита = P5 (структурный аналог).
COULOMB_K = 3.6e-4       # Постоянная Кулона (модельные единицы)
COULOMB_SOFT = 0.35      # Сглаживание Платтера ε (ячейки): V = −K·q₁q₂/√(d²+ε²)
COULOMB_SOFT2 = COULOMB_SOFT ** 2
EM_DT = 1.0              # Шаг интегрирования (симплициев Евлер)
EM_VMAX = 0.95           # Ограничение скорости (ячейки/шаг): ≤ 1 ячейки за шаг
HADRON_RADIUS = 2        # Радиус хадронизации: макс. попарное расстояние кварков (ячейки)

# --- L3b: ДАЛЬНЯЯ ГРАВИТАЦИЯ (поле Пуассона на торе, FFT) ---
# Флаг long_range_gravity (дефолт False: контактный 1-клеточный шаблон,
# фазы 1-8 бит-идентичны). С флагом: g = g_контакт + g_Пуассон, где
# ∇²g = -4πG·(ρ − ρ̄), k=0 → 0 (условие разрешимости на торе).
# POISSON_4PI_G — константа G в узельных единицах (свободный параметр, п.1).
POISSON_4PI_G = 0.04

# --- МАСШТАБ МАСС: 1 единица энергии узла = 1.2 = масса протона (PDG) ---
NODE_MASS_SCALE = 1.2 / 938.27208816e6   # модельных единиц на эВ


def _hadron_release(hsym: str) -> float:
    """Массовый избыток адрона над кварками (PDG), в модельных единицах.
    Для p: 929 МэВ ≈ 99% массы — энергия QCD-поля (конфайнмент)."""
    h = _CATALOG[hsym]
    qsum = sum(_CATALOG[c].mass_eV
               for c in HADRON_COMPOSITION[hsym] if c in _CATALOG)
    return float(h.mass_eV - qsum) * NODE_MASS_SCALE


HADRON_RELEASE = {"p": _hadron_release("p"), "n": _hadron_release("n")}

# --- S2 (DESIGN_S2 §5): единое ядро — ОДИН масштаб массы (принцип
# эквивалентности): m_dyn := M_ledger, протон = 1.2 модельных единиц
# (NODE_MASS_SCALE). Legacy-таблица _NODE_MEFF (протон = 1.0) сохранена для
# legacy-ядра (бит-закреп: фазы 1–8, 76 сверок, κ D12).
def _node_meff_uni(sym: str) -> float:
    """Масса узла в модельных единицах через M_ledger (NODE_MASS_SCALE):"""
    p = _particle(sym)
    if p is None or p.mass_eV <= 0:
        return 0.0
    return p.mass_eV * NODE_MASS_SCALE


_NODE_MEFF_UNI = {s: _node_meff_uni(s) for s in _NODE_TYPES}

# --- МАСШТАБНАЯ КАРТА СИ (THEORY.md §4, S1): 2 якоря, ПРОИЗВОДНЫЕ, не диалы ---
# якорь 1 (энергия): U = 938.27208816/1.2 МэВ (хадронизация, NODE_MASS_SCALE);
# якорь 2 (длина): alpha-якорь K·U·h = alpha·ħ·c (K = 3.6e-4, S1 — THEORY.md
#    §4, F4; при этом K круговая кулонова орбита стабильна — D18).
# Самопроверка: 1.2·SI_U_J/C² = 1.672622e-27 кг = M_P (до 1e-8).
from .constants import (ALPHA as _SI_ALPHA, C as _SI_C, E_CH as _SI_E_CH,
                        HBAR as _SI_HBAR, M_P as _SI_M_P)

SI_U_MEV = 938.27208816 / 1.2                     # 1 модельная ед. энергии, МэВ
SI_U_J = SI_U_MEV * 1e6 * _SI_E_CH                # 1 модельная ед. энергии, J
SI_H_M = _SI_ALPHA * _SI_HBAR * _SI_C / (COULOMB_K * SI_U_J)  # 1 ячейка, м
SI_DT_S = SI_H_M / _SI_C                          # 1 шаг, с
# (1 шаг = 20.2704·(ħ/U); физ. G в модельных ед. = 4πG_phys = 2.54e-39,
#  кросс-чек: g(R_Земли) = GM/(R·c²) = 6.96e-10; модельное 4πG = POISSON_4PI_G
#  = 0.04 ⇒ модельная G = 1.57e37 × физ. — S7/G1: масштабная граница, S7/аудит G3)

# --- ЭНТРОПИЙНЫЙ РЕГУЛЯТОР ---
ENTROPY_TARGET = 0.35        # Целевая энтропия (динамическое равновесие)
ENTROPY_PRESSURE = 0.02      # Сила давления регулятора
ENTROPY_RESPONSE_TIME = 10   # Период проверки (каждых N шагов)

TWO_PI = 2 * np.pi


# ============================================================
# 3D ОБЪЁМ ИЗ 5 СУБПРОСТРАНСТВ
# ============================================================

class Volume3D:
    """
    Трёхмерный объём (тор: границы замкнуты).
    Каждая ячейка несёт состояние во всех 5 субпространствах.
    """

    def __init__(self, size: int = GRID_SIZE, seed: int = 42,
                 unified_action: bool = False):
        self.size = size
        self.n = size ** 3
        self.shape = (size, size, size)
        self.rng = np.random.default_rng(seed)
        self.time_step: int = 0

        # --- L0: ВРЕМЯ ---
        # Фаза времени [0, 2*pi] — где ячейка в «цикле»
        self.time_phase = self.rng.uniform(0, TWO_PI, self.n)
        # Локальная скорость течения времени [0.3, 1.5]
        self.time_rate = np.ones(self.n)

        # --- L1: ЭНЕРГИЯ ---
        self.energy = ENERGY_ZERO_POINT + self.rng.normal(0, 0.003, self.n)

        # --- L2: ВЕРОЯТНОСТЬ ---
        # Комплексная волновая функция psi = real + i*imag
        self.prob_real = self.rng.normal(0, 0.02, self.n)
        self.prob_imag = self.rng.normal(0, 0.02, self.n)

        # --- L3: ГРАВИТАЦИЯ ---
        # Гравитационный потенциал [-1, 0] (отрицательный = яма)
        self.grav = np.zeros(self.n)

        # --- L4: ПОЛЯ ---
        # Вектор поля (3 компоненты), [-1, 1]
        self.field = self.rng.normal(0, 0.01, (3,) + self.shape)
        # «Заряд» (источник поля) [-1, 1]
        self.charge = np.zeros(self.n)

        # --- Метрики ---
        self.entropy_local = np.zeros(self.n)
        self.matter = np.zeros(self.n, dtype=bool)
        self.matter_age = np.zeros(self.n, dtype=np.int32)
        self.particle_type = np.zeros(self.n, dtype=np.int32)  # код типа (_NODE_TYPES), по умолчанию "n"

        # --- Спин и 2 слота на ячейку (принцип Pauli) ---
        # Слот 1 = спин ↑ (1), слот 2 = спин ↓ (0); доп. бозоны —
        # stack_extra (≤ BOSE_STACK_MAX). matter/particle_type/matter_age/
        # stack — производные виды (_sync_views).
        self.node_t = np.full(self.n, -1, dtype=np.int16)      # код типа, слот 1
        self.node_t2 = np.full(self.n, -1, dtype=np.int16)     # код типа, слот 2
        self.spin = np.full(self.n, -1, dtype=np.int8)         # спин, слот 1
        self.spin2 = np.full(self.n, -1, dtype=np.int8)        # спин, слот 2
        self.matter_age2 = np.zeros(self.n, dtype=np.int32)    # возраст, слот 2
        self.stack_extra = np.zeros(self.n, dtype=np.int8)     # бозоны сверх 2 слотов
        self.node_origin = np.full(self.n, -1, dtype=np.int32)  # id события создания (цепочка)
        self.node_origin2 = np.full(self.n, -1, dtype=np.int32)
        self.stack = np.zeros(self.n, dtype=np.int8)           # всего узлов в ячейке (0..3)
        # Струны (конфайнмент): {"a": idx, "b": idx, "born": t}
        self.strings: list = []
        # События динамики: {"t","id","parent","pos","from","to","depth","type"}
        self.decay_events: list = []
        self._events_by_id: dict = {}
        self._decay_id: int = 0
        # Статистика (демо)
        self.pauli_blocks: int = 0       # Pauli: заблокированные фермионы
        self.bose_stacks: int = 0        # размещённые доп. бозоны (стек)
        self.n_quarks_created: int = 0
        self.n_strings_broken: int = 0
        self.n_hadronized: int = 0       # хадронизаций (uud→p, udd→n)

        # --- ЭМ-динамика (фазы 10–11): кулон заряженных узлов ---
        # По умолчанию ВЫКЛЕНА: старый сценарий (фазы 1–8) бит-идентичен.
        self.em_dynamics: bool = False
        # --- Дальняя гравитация (поле Пуассона, флаг; п.0) ---
        # Дефолт False: контактный шаблон (бит-идентично старым сценариям).
        self.long_range_gravity: bool = False
        # --- S2 (DESIGN_S2): флаг единого ядра ---
        # False = legacy (бит-закреп: фазы 1–8, 76 сверок, κ D12).
        # True = step_unified: единый масштаб массы (F1: m_dyn := M_ledger)
        #   + строго юнитарное ядро ψ (SSFM): Нётер U(1)/p — структурно.
        self.unified_action: bool = unified_action
        self._mef = _NODE_MEFF_UNI if unified_action else _NODE_MEFF
        # Непрерывные позиции узлов слотов (координаты ячеек; при создании = ячейка)
        _xs = (np.arange(self.n) // (self.size * self.size)).astype(np.float64)
        _ys = ((np.arange(self.n) // self.size) % self.size).astype(np.float64)
        _zs = (np.arange(self.n) % self.size).astype(np.float64)
        self.pos_x, self.pos_y, self.pos_z = _xs, _ys, _zs
        self.pos2_x, self.pos2_y, self.pos2_z = _xs.copy(), _ys.copy(), _zs.copy()
        # Скорости узлов слотов (ячейки/шаг); 0, когда слот пуст
        self.vel_x = np.zeros(self.n)
        self.vel_y = np.zeros(self.n)
        self.vel_z = np.zeros(self.n)
        self.vel2_x = np.zeros(self.n)
        self.vel2_y = np.zeros(self.n)
        self.vel2_z = np.zeros(self.n)

        # Кэшированные производные
        self.psi_abs = np.zeros(self.n)
        self.psi_phase = np.zeros(self.n)
        self.field_mag = np.zeros(self.n)

        # --- Глобальные метрики ---
        self.global_entropy: float = 0.0
        self.total_energy: float = 0.0
        self.matter_count: int = 0
        self.entropy_history: list = []
        self.energy_history: list = []
        self.matter_history: list = []
        self.layer_stats_history: list = []

        # --- Энтропийный регулятор ---
        self.entropy_regulator_active: bool = True
        self.entropy_pressure_current: float = 0.0
        self._entropy_counter: int = 0

        # --- Энергетический ledger: учёт ВСЕХ явных источников/стоков ΣE ---
        # Свободная часть схемы (перетекание, гравитационный член, казимировский
        # пол) консервативна: потоки попарно-антисимметричны, Σ=0 на торе, clamp
        # с нулевым нетто. Явные источники (рождение/распад/пинк/регулятор/
        # впрыски/сила) — детерминированная «внешняя работа»; тождество
        # ΔΣE(T) = ledger(T) проверяется до машинной точности (experiments.py D9).
        self.energy_ledger: float = 0.0
        self.energy_ledger_detail: dict = {}

        self._init_vacuum()

    # --------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ ВАКУУМОМ (раслоение)
    # --------------------------------------------------------

    def _init_vacuum(self):
        """Вакуум = раслоение: слои существуют, но не синтезированы."""
        # Фазы уже случайны (суперпозиция), энергия = нулевая точка + шум
        self.energy = np.maximum(self.energy, ENERGY_ZERO_POINT)
        self._sync_views()
        self._recompute_derived()

    # --------------------------------------------------------
    # ВЕКТОРНЫЕ ВСПОМОГАТЕЛИ
    # --------------------------------------------------------

    @staticmethod
    def _lap3(a: np.ndarray) -> np.ndarray:
        """Лапласиан на торе (6 соседей) по 3 последним осям."""
        nd = a.ndim
        a0, a1, a2 = nd - 3, nd - 2, nd - 1
        return (
            np.roll(a, 1, a0) + np.roll(a, -1, a0)
            + np.roll(a, 1, a1) + np.roll(a, -1, a1)
            + np.roll(a, 1, a2) + np.roll(a, -1, a2)
        ) - 6.0 * a

    def _lap(self, a: np.ndarray) -> np.ndarray:
        """Лапласиан; принимает плоский (n,) или 3D (s,s,s) массив."""
        if a.ndim == 1:
            return self._lap3(a.reshape(self.shape)).flatten()
        return self._lap3(a)

    def _neighbor_avg(self, a: np.ndarray) -> np.ndarray:
        """Среднее по 6 ближайшим соседям (laplacean + 6a)/6."""
        return (self._lap(a) + 6.0 * a) / 6.0

    def _roll_far(self, a: np.ndarray) -> np.ndarray:
        """Среднее по 6 «дальним» соседям (радиус 2) — нелокальность."""
        a3 = a.reshape(self.shape)
        return (
            np.roll(a3, 2, 0) + np.roll(a3, -2, 0)
            + np.roll(a3, 2, 1) + np.roll(a3, -2, 1)
            + np.roll(a3, 2, 2) + np.roll(a3, -2, 2)
        ).flatten() / 6.0

    def _flat(self, x: int, y: int, z: int) -> int:
        """Плоский индекс ячейки (порядок C: [x][y][z])."""
        s = self.size
        return (x * s + y) * s + z

    def _led(self, key: str, d: float):
        """Занести в ledger явное изменение ΣE (источник/сток)."""
        if d:
            self.energy_ledger += float(d)
            self.energy_ledger_detail[key] = self.energy_ledger_detail.get(key, 0.0) + float(d)

    def _recompute_derived(self):
        r, s = self.prob_real, self.prob_imag
        self.psi_abs = np.sqrt(r * r + s * s)
        self.psi_phase = np.arctan2(s, r)
        self.field_mag = np.sqrt(np.sum(self.field ** 2, axis=0)).flatten()

    def _layer_values(self, layer: int) -> np.ndarray:
        if layer == TIME:
            return np.sin(self.time_phase)
        if layer == ENERGY:
            return self.energy
        if layer == PROBABILITY:
            return self.psi_abs
        if layer == GRAVITY:
            return self.grav
        return self.field_mag

    # --------------------------------------------------------
    # ГЛАВНЫЙ ШАГ ЭВОЛЮЦИИ
    # --------------------------------------------------------

    def step(self, external_injection: dict = None):
        """Один шаг эволюции (legacy-ядро; порядок слоёв = иерархия причинности).
        При unified_action=True — единое ядро step_unified() (DESIGN_S2)."""
        if self.unified_action:
            return self.step_unified(external_injection=external_injection)
        self.time_step += 1

        # 1. ГРАВИТАЦИЯ: кривизна из плотности энергии (Эйнштейн)
        self._update_gravity()
        # 2. ВРЕМЯ: локальная скорость течения (ОТО + СТО)
        self._update_time()
        # 3. ЭНЕРГИЯ: перетекание, консервация
        self._update_energy()
        # 4. ВЕРОЯТНОСТЬ: уравнение Шрёдингера (упрощённое)
        self._update_probability()
        # 5. ПОЛЯ: уравнения Максвелла (упрощённые)
        self._update_fields()
        # 5.5. ЭМ-ДИНАМИКА: кулон заряженных узлов (гейт em_dynamics)
        self._update_em()
        # 6. ВЗАИМОДЕЙСТВИЯ между слоями
        self._update_interactions()
        # 7. ФАЗОВЫЕ ПЕРЕХОДЫ (материя <-> вакуум; цепочки распадов)
        self._phase_transitions()
        # 7.5. КВАРКОВЫЕ СТРУНЫ (конфайнмент: σ·d, стягивание/разрыв)
        self._update_strings()
        # 8. ЭНТРОПИЙНЫЙ РЕГУЛЯТОР
        self._entropy_regulator()
        # 9. Глобальные метрики
        self._update_metrics()
        # 10. Внешнее воздействие
        if external_injection:
            self._apply_injection(external_injection)

    def step_unified(self, external_injection: dict = None):
        """Единое ядро (S2, DESIGN_S2.md): сборка единого дискретного действия.

        Действие S = Σ_T (L_E + L_ψ + L_F + L_G) + S_узлы:
          L_E  — симметричные потоки энергии (L1): ΣE консервативно (Нётер T);
          L_ψ  — юнитарное ядро SSFM, i∂tψ = (−½∇² + V)ψ, V = 2g + E (L2):
                 Нётер U(1) (N = Σ|ψ|²) и p сохраняются строго (дрейф ~1e-15);
          L_F  — дипольное поле (L4): калибровка, экранирование Дебая;
          L_G  — гауссова связка g = −∇²ρ (L3, FFT-Poisson — точно);
          S_узлы — −m∫dt + V_кулон: единая масса m_dyn := M_ledger (F1).
        Связки:
          rate — метрическая (dτ/dt = 1 + κ_g·g + κ_E·(E−Ē) + κ_v·v²);
          g    — гауссова (нет динамики: g вычисляется из ρ).
        Явные внешние члены (НЕ входят в действие; учтены в ledger):
          нелокальность (Белл), шум Гейзенберга, pin-узлы, межслойные
          взаимодействия, рождения/распады, энтропийный регулятор, впрыски.

        Отличия от legacy: (a) m_dyn := M_ledger (протон = 1.2, F1);
        (b) юнитарное ядро ψ (SSFM) вместо явной схемы. Порядок обновлений —
        тот же (иерархия причинности).
        """
        self.time_step += 1
        # 1. ГРАВИТАЦИЯ: гауссова связка (g из ρ)
        self._update_gravity()
        # 2. ВРЕМЯ: метрическая связка
        self._update_time()
        # 3. ЭНЕРГИЯ: симметричные потоки (Нётер T)
        self._update_energy()
        # 4. ВЕРОЯТНОСТЬ: юнитарное ядро SSFM + явные внешние члены
        self._update_probability_uni()
        # 5. ПОЛЯ: дипольное поле (калибровка, Дебай)
        self._update_fields()
        # 5.5. ЭМ-ДИНАМИКА: кулон (единая масса)
        self._update_em()
        # 6. ВЗАИМОДЕЙСТВИЯ между слоями (явные, ledger)
        self._update_interactions()
        # 7. ФАЗОВЫЕ ПЕРЕХОДЫ (внешние события, ledger)
        self._phase_transitions()
        # 7.5. КВАРКОВЫЕ СТРУНЫ (конфайнмент)
        self._update_strings()
        # 8. ЭНТРОПИЙНЫЙ РЕГУЛЯТОР (термостат, ledger)
        self._entropy_regulator()
        # 9. Глобальные метрики
        self._update_metrics()
        # 10. Внешнее воздействие
        if external_injection:
            self._apply_injection(external_injection)

    def _update_probability_uni(self):
        """L2 (unified-ядро): строго юнитарное ядро (SSFM) + явные внешние члены.

        Ядро: i·∂ψ/∂t = (−½∇² + V)ψ, V = 2g + E (стандартные знаки,
        DESIGN_S2 §3; legacy-схема — комплексно-сопряжённая форма той же
        юнитарной динамики).
        - Кинетический член диагонален в Фурье (тор, roll-лапласиан):
          λ_k = −4·Σ_d sin²(π·k_d/N_d) — собственные значения ∇²;
          ψ̂_k → exp(i·½·λ_k·dt)·ψ̂_k (точно: фазовый поворот, нет диссипации).
        - Потенциал: ψ → exp(−i·V·dt/2)·ψ (полупошаг, ×2).
        SSFM юнитарен → N = Σ|ψ|² и решёточный импульс сохраняются строго
        (дрейф ~1e-15, а не O(dt²) явной схемы): Нётер U(1)/сдвиги —
        структурное свойство ядра, не свойство схемы.
        Явные внешние члены (нелокальность Белла, шум Гейзенберга, pin-узлы,
        clamp) — идентичны legacy: внешние возмущения, не часть ядра
        (THEORY.md A6, DESIGN_S2 §3.4).
        """
        r, s = self.prob_real, self.prob_imag
        V = self.grav * 2.0 + self.energy
        dt = PROB_INTERFERENCE * 0.03

        # Потенциальный полупошаг: ψ → e^{−iVdt/2}ψ (фазовый поворот)
        a = 0.5 * dt * V
        c, sn = np.cos(a), np.sin(a)
        r1 = r * c + s * sn
        s1 = -r * sn + s * c

        # Кинетический полный шаг в Фурье-пространстве: ψ̂ → e^{iλdt/2}·ψ̂
        kx = np.fft.fftfreq(self.size)
        lam1d = -4.0 * np.sin(np.pi * kx) ** 2
        lam = (lam1d[:, None, None] + lam1d[None, :, None]
               + lam1d[None, None, :])
        ps = np.fft.fftn((r1 + 1j * s1).reshape(self.shape))
        ps *= np.exp(0.5j * dt * lam)
        out = np.fft.ifftn(ps)
        r2, s2 = out.real.reshape(-1), out.imag.reshape(-1)

        # Второй потенциальный полупошаг (временные: оба компонента из старых)
        r2n = r2 * c + s2 * sn
        s2n = -r2 * sn + s2 * c
        r2, s2 = r2n, s2n

        # --- Явные внешние члены (идентичны legacy) ---
        # Нелокальность: корреляция с «дальними» соседями (радиус 2)
        r2 = r2 + PROB_NONLOCALITY * 0.01 * (self._roll_far(r2) - r2)
        s2 = s2 + PROB_NONLOCALITY * 0.01 * (self._roll_far(s2) - s2)
        # Неопределённость: шум Гейзенберга
        r2 = r2 + self.rng.normal(0, PROB_UNCERTAINTY * 0.02, self.n)
        s2 = s2 + self.rng.normal(0, PROB_UNCERTAINTY * 0.02, self.n)
        # Связанное состояние: узел «пинает» амплитуду к PSI_NODE_AMP
        amp = np.sqrt(r2 * r2 + s2 * s2)
        pin = self.matter & (amp > 1e-6)
        if pin.any():
            pull = 0.05 * (PSI_NODE_AMP - amp)
            inv = np.where(amp > 1e-12, 1.0 / np.maximum(amp, 1e-12), 0.0)
            r2 = r2 + np.where(pin, pull * r2 * inv, 0.0)
            s2 = s2 + np.where(pin, pull * s2 * inv, 0.0)
        # Нормировка: |psi| <= 1
        over = amp > 1.0
        if over.any():
            r2 = np.where(over, r2 / np.maximum(amp, 1e-12), r2)
            s2 = np.where(over, s2 / np.maximum(amp, 1e-12), s2)

        self.prob_real, self.prob_imag = r2, s2

    # --------------------------------------------------------
    # L3: ГРАВИТАЦИЯ (LIGO 2015, GPS)
    # --------------------------------------------------------

    def _update_gravity(self):
        """
        Кривизна = f(плотность энергии).
        Аналог: G_{mu,nu} = 8*pi*G * T_{mu,nu}
        Упрощение: grav = -G * (own + <соседи>)
        """
        # Источник: плотность энергии + СОБСТВЕННАЯ МАССА узлов (E=mc^2):
        # материя гравитирует всегда, даже «остывшая» — её яма притягивает
        # энергию обратно (самоподдержание узла).
        src = np.clip(self.energy - ENERGY_ZERO_POINT, 0.0, None)
        src = src + MATTER_REST_MASS * self.matter.astype(float)
        potential = -GRAV_STRENGTH * (src + self._neighbor_avg(src))
        if self.long_range_gravity:
            potential = potential + self._poisson_field(src)
        self.grav = np.clip(potential, -1.0, 0.0)

    def _poisson_field(self, src: np.ndarray) -> np.ndarray:
        """Точное решение ∇²g = -4πG·(src − mean src) на торе (FFT).

        k=0 → 0 (нулевое среднее источника — условие разрешимости).
        Дистанционное поле: g(r) ≈ -G·M/r вдали от источника
        (G = POISSON_4PI_G / 4π). Безынерционное (ньютоновский предел;
        волновые поправки порядка v²/c² — за рамками модели)."""
        rho = (src - src.mean()).reshape(self.shape)
        rf = np.fft.rfftn(rho)
        k2 = getattr(self, "_poisson_k2", None)
        if k2 is None:                                    # сетка зависит только от size
            k = 2.0 * np.pi
            k0 = k * np.fft.fftfreq(self.size)            # полная (оси 0, 1)
            self._poisson_k2 = ((k0[:, None, None] ** 2 + k0[None, :, None] ** 2
                                 + (k * np.fft.rfftfreq(self.size))[None, None, :] ** 2))
            k2 = self._poisson_k2
        gf = -POISSON_4PI_G * rf / np.where(k2 == 0.0, 1.0, k2)
        gf[0, 0, 0] = 0.0
        return np.fft.irfftn(gf, s=self.shape, axes=(0, 1, 2)).ravel()

    # --------------------------------------------------------
    # L0: ВРЕМЯ (Hafele-Keating 1971, мюоны)
    # --------------------------------------------------------

    def _update_time(self):
        """
        Локальная скорость времени:
        - замедляется в гравитационной яме (ОТО): t' = t * (1 + 2*Phi/c^2)^~1/2
        - замедляется при высокой энергии (СТО):  t' = t * sqrt(1 - v^2/c^2)
        """
        grav_factor = 1.0 + TIME_DILATION_GRAV * self.grav
        vel_factor = 1.0 - TIME_DILATION_VEL * np.clip(self.energy, 0.0, 1.0)
        self.time_rate = np.clip(grav_factor * vel_factor, 0.3, 1.5)
        self.time_phase = (self.time_phase + TIME_FLOW * self.time_rate * 0.1) % TWO_PI

    # --------------------------------------------------------
    # L1: ЭНЕРГИЯ (консервация Нётер, E=mc^2, нулевая точка)
    # --------------------------------------------------------

    def _update_energy(self):
        """
        Перетекание: из высокой плотности -> низкой (диффузия).
        УЗЛЫ МАТЕРИИ ЗАПЕРТЫ: в ячейках-узлах диффузия слабее (замкнутая
        комбинация связностей удерживает энергию) — скорость симметрична по
        паре ячеек, поэтому ΣE сохраняется.
        Плюс: энергия стекает в гравитационные ямы (масса кривит пространство,
        пространство «тянет» энергию). Поток консервативен: на торе
        Σ(neighbor_avg(g) - g) = 0.
        Нулевая энергия: E >= E_0 (Казимир).
        """
        e = self.energy
        mf = self.matter.astype(float)
        m3 = mf.reshape(self.shape)
        e3 = e.reshape(self.shape)

        base = self._lap(e)  # Σ(e_n - e) по 6 соседям
        msum = np.zeros(self.shape)
        esum = np.zeros(self.shape)
        for ax in range(3):
            for s in (-1, 1):
                msum += np.roll(m3, s, ax)
                esum += np.roll(m3 * e3, s, ax)
        # Сумма (e_n - e) только по «связанным» рёбрам (сосед — узел):
        # для узла все 6 рёбер связаны, для вакуума — только с узлами.
        red = np.where(m3 > 0, base.reshape(self.shape), esum - msum * e3).flatten()
        # Симметричная по паре скорость: flow_edge = c0*(1 - B*max(m_c, m_n)),
        # поэтому суммарный обмен по всем парам = 0 (ΣE сохраняется строго).
        flow = ENERGY_FLOW_RATE * (base - NODE_BIND_STRENGTH * red)

        e = e + flow + GRAV_ENERGY_COUPLING * (
            self._neighbor_avg(self.grav) - self.grav
        )

        # Казимировский пол E >= E0: дефицит равномерно всасывается всем объёмом
        # (иначе clamp создавал бы энергию), ΣE остаётся без изменения.
        created = float((ENERGY_ZERO_POINT - e)[e < ENERGY_ZERO_POINT].sum())
        e = np.maximum(e, ENERGY_ZERO_POINT)
        if created > 0.0:
            e -= created / self.n
        self.energy = e

    # --------------------------------------------------------
    # L2: ВЕРОЯТНОСТЬ (двойная щель, Белл, Гейзенберг)
    # --------------------------------------------------------

    def _update_probability(self):
        """
        Упрощённое уравнение Шрёдингера:
        i*hbar * d(psi)/dt = -hbar^2/2m * lap(psi) + V * psi
        для psi = r + i*s (и d(psi)/dt = i*(-0.5*lap(psi) + V*psi)):
            dr/dt =  0.5 * lap(s) - V * s
            ds/dt = -0.5 * lap(r) + V * r
        (знаки выбраны так, чтобы схема была унитарной: норма |psi|^2 сохраняется)
        Потенциал V = 2*grav + energy (гравитация и энергия гнут вероятность).
        Плюс: нелокальность (Белл) и неопределённость (шум Гейзенберга).
        """
        r, s = self.prob_real, self.prob_imag
        V = self.grav * 2.0 + self.energy
        dt = PROB_INTERFERENCE * 0.03

        nr = r + dt * (0.5 * self._lap(s) - V * s)
        ns = s + dt * (-0.5 * self._lap(r) + V * r)

        # Нелокальность: корреляция с «дальними» соседями (радиус 2)
        nr = nr + PROB_NONLOCALITY * 0.01 * (self._roll_far(nr) - nr)

        # Неопределённость: шум Гейзенберга
        nr = nr + self.rng.normal(0, PROB_UNCERTAINTY * 0.02, self.n)
        ns = ns + self.rng.normal(0, PROB_UNCERTAINTY * 0.02, self.n)

        # Связанное состояние: узел «пинает» амплитуду psi к PSI_NODE_AMP
        # (дискретный спектр атома: узел удерживает свою волну)
        amp = np.sqrt(nr * nr + ns * ns)
        pin = self.matter & (amp > 1e-6)
        if pin.any():
            pull = 0.05 * (PSI_NODE_AMP - amp)
            # Безопасное деление: в вакуумных ячейках amp может быть строго 0
            # (np.where вычисляет оба ветвления) -> без 0/0, семантика в узлах
            # не меняется (там amp > 1e-6 > 1e-12).
            inv = np.where(amp > 1e-12, 1.0 / np.maximum(amp, 1e-12), 0.0)
            nr = nr + np.where(pin, pull * nr * inv, 0.0)
            ns = ns + np.where(pin, pull * ns * inv, 0.0)

        # Нормировка: |psi| <= 1
        over = amp > 1.0
        if over.any():
            nr = np.where(over, nr / np.maximum(amp, 1e-12), nr)
            ns = np.where(over, ns / np.maximum(amp, 1e-12), ns)

        self.prob_real, self.prob_imag = nr, ns

    # --------------------------------------------------------
    # L4: ПОЛЯ (Максвелл, Штерн-Герлах, U(1))
    # --------------------------------------------------------

    def _update_fields(self):
        """
        Упрощённые уравнения Максвелла:
        - Поле релаксирует к градиенту заряда (Gauss: div E = rho/eps_0),
          дипольное приближение E ~ -grad(charge)
        - Волновое распространение: малая дисперсия (скорость = c)
        - Калибровочная симметрия: «перенормировка» амплитуды
        - Заряд релаксирует к 0 (экранирование Дебая)
        """
        k = FIELD_STRENGTH
        c3 = self.charge.reshape(self.shape)
        target = np.stack([
            k * (np.roll(c3, -1, 0) - np.roll(c3, 1, 0)),
            k * (np.roll(c3, -1, 1) - np.roll(c3, 1, 1)),
            k * (np.roll(c3, -1, 2) - np.roll(c3, 1, 2)),
        ])  # (3, s, s, s)
        f = self.field + 0.02 * self._lap3(self.field)   # волна (по 3D-блокам)
        f = f + 0.3 * (target - f)                       # релаксация
        f = f * (1.0 - FIELD_GAUGE * 0.1)                # калибровка
        self.field = f
        self.charge *= 0.999

    # --------------------------------------------------------
    # ВЗАИМОДЕЙСТВИЯ МЕЖДУ СЛОЯМИ
    # --------------------------------------------------------

    def _update_interactions(self):
        """
        Кросс-слойные связи (из физики):
        - Поля -> Вероятность  (QED: фотон меняет состояние электрона)
        - Вероятность -> Энергия (квантовые флуктуации)
        - Гравитация -> Поля  (красное смещение)
        - Время модулирует энергию (осцилляция)
        """
        # 1. Поля влияют на вероятность (QED)
        self.prob_real += self.field_mag * 0.01 * np.sin(self.time_phase)
        # 2. Вероятность влияет на энергию (флуктуации; слабый канал,
        #    чтобы ΣE не дрейфовала: основной «нагрев» — регулятор/тёмная энергия)
        self._led("psi2_heat", float(((self.prob_real ** 2 + self.prob_imag ** 2) * 0.0005).sum()))
        self.energy += (self.prob_real ** 2 + self.prob_imag ** 2) * 0.0005
        # 3. Гравитация влияет на поля (красное смещение)
        self.field *= (1.0 + self.grav * 0.01).reshape(1, *self.shape)
        # 4. Время модулирует всё (осцилляция)
        self._led("time_mod", float((np.sin(self.time_phase) * self.energy).sum()) * 0.001)
        self.energy += np.sin(self.time_phase) * 0.001 * self.energy

    # --------------------------------------------------------
    # ФАЗОВЫЕ ПЕРЕХОДЫ (материя <-> вакуум)
    # --------------------------------------------------------

    def _phase_transitions(self):
        """
        Материя = «узел», где все 5 слоёв коррелированы (синтез).
        Вакуум = раслоение (низкая корреляция).
        Рождение = скачок энергии (кристаллизация узла) — по спин-слотам (Pauli).
        Распад = энергия -> тепло; для τ/μ/π — цепочка (дочерний узел).
        """
        states = np.stack([
            np.sin(self.time_phase),
            self.energy / (self.energy + 0.1),
            self._psi_abs_now(),
            -self.grav,
            self._field_mag_now(),
        ])
        synthesis = np.mean(np.abs(states), axis=0)

        occ1 = self.node_t >= 0
        occ2 = self.node_t2 >= 0

        # Lifetime по типу: стабильные типы (None) не распадаются
        life1 = np.full(self.n, np.inf, dtype=np.float64)
        life2 = life1.copy()
        for code, sym in enumerate(_NODE_TYPES):
            tl = TYPE_LIFETIME[sym]
            if tl is not None:
                life1[self.node_t == code] = float(tl)
                life2[self.node_t2 == code] = float(tl)
        # Распад: либо глубокая декогеренция (синтез < 0.15), либо возраст.
        # Порог низкий: сам узел (запёртый psi) держит синтез выше ~0.17.
        dec1 = occ1 & ((self.matter_age >= life1) | (synthesis < 0.15))
        dec2 = occ2 & ((self.matter_age2 >= life2) | (synthesis < 0.15))
        decayed = dec1 | dec2

        # Распад: дочерний (цепочка) или только энергия; событие -> decay_events
        for idx in np.where(decayed)[0]:
            self._decay_node(int(idx), 1 if dec1[idx] else 2)

        # Очистка распавшихся слотов
        for arr, mask in ((self.node_t, dec1), (self.node_t2, dec2),
                          (self.spin, dec1), (self.spin2, dec2),
                          (self.node_origin, dec1), (self.node_origin2, dec2)):
            arr[mask] = -1
        self.matter_age[dec1] = 0
        self.matter_age2[dec2] = 0

        # Рождение: спонтанная кристаллизация в свободной ячейке (демо: "n")
        born_mask = (synthesis > 0.4) & ~occ1 & ~occ2 & (self.stack_extra == 0)
        for idx in np.where(born_mask)[0]:
            self._place_node(int(idx), TYPE_CODE["n"], int(self.rng.integers(2)),
                             energize=False)

        # Кристаллизация: выделение энергии; распад: энергия -> тепло,
        # вероятность «размазывается»
        self._led("birth", 0.1 * float(born_mask.sum()))
        self.energy += 0.1 * born_mask
        self._led("decay", 0.05 * float(decayed.sum()))
        self.energy += 0.05 * decayed
        damp = np.where(decayed, 0.5, 1.0)
        self.prob_real *= damp
        self.prob_imag *= damp
        # Возраст выживших узлов
        self.matter_age = np.where(self.node_t >= 0, self.matter_age + 1, 0).astype(np.int32)
        self.matter_age2 = np.where(self.node_t2 >= 0, self.matter_age2 + 1, 0).astype(np.int32)

        self._sync_views()

    # --------------------------------------------------------
    # ДИНАМИКА ЧАСТИЦ: спин, Pauli/Бозе, цепочки распадов, струны
    # --------------------------------------------------------

    def _sync_views(self):
        """matter/particle_type/matter_age/stack — производные виды слотов."""
        occ1 = self.node_t >= 0
        occ2 = self.node_t2 >= 0
        self.matter = occ1 | occ2
        self.particle_type = np.where(
            occ1, self.node_t, np.where(occ2, self.node_t2, 0)).astype(np.int32)
        self.matter_age = np.where(
            occ1, self.matter_age, np.where(occ2, self.matter_age2, 0)).astype(np.int32)
        self.stack = occ1.astype(np.int8) + occ2.astype(np.int8) + self.stack_extra
        self.matter_count = int(self.stack.sum())

    def _set_slot(self, idx: int, slot: int, code: int, spin: int,
                  origin: int = -1, energize: bool = True, age: int = 0):
        """Записать узел в слот (вызыватель гарантирует, что слот свободен)."""
        if slot == 1:
            self.node_t[idx] = code
            self.spin[idx] = spin
            self.matter_age[idx] = age
            self.node_origin[idx] = origin
            self.pos_x[idx] = idx // (self.size * self.size)
            self.pos_y[idx] = (idx // self.size) % self.size
            self.pos_z[idx] = idx % self.size
            self.vel_x[idx] = 0.0
            self.vel_y[idx] = 0.0
            self.vel_z[idx] = 0.0
        else:
            self.node_t2[idx] = code
            self.spin2[idx] = spin
            self.matter_age2[idx] = age
            self.node_origin2[idx] = origin
            self.pos2_x[idx] = idx // (self.size * self.size)
            self.pos2_y[idx] = (idx // self.size) % self.size
            self.pos2_z[idx] = idx % self.size
            self.vel2_x[idx] = 0.0
            self.vel2_y[idx] = 0.0
            self.vel2_z[idx] = 0.0
        if energize:
            # Кристаллизация: состояние узла (запёртый psi, заряд, пинк поля)
            if float(self.energy[idx]) < 1.2:
                self._led("pin", 1.2 - float(self.energy[idx]))
                self.energy[idx] = 1.2
            self.prob_real[idx] = 0.7
            self.prob_imag[idx] = 0.0
            self.charge[idx] = float(self.rng.choice([-1, 1]) * 0.5)
            x = idx // (self.size * self.size)
            y = (idx // self.size) % self.size
            z = idx % self.size
            self.field[:, x, y, z] = [0.3, -0.2, 0.4]
        self._sync_views()

    def _place_node(self, idx: int, code: int, spin: int, origin: int = -1,
                    energize: bool = True) -> bool:
        """Разместить узел по правилам Ферми/Бозе. False = Pauli-блок/ячейка полна.
        Фермион (спин 1/2) -> только в свободный слот своего спина;
        бозон -> любой свободный слот, либо стек (≤ BOSE_STACK_MAX)."""
        sym = _NODE_TYPES[code]
        if sym in FERMI_TYPES:
            if spin == 1:
                if self.node_t[idx] < 0:
                    self._set_slot(idx, 1, code, spin, origin, energize)
                    return True
            elif self.node_t2[idx] < 0:
                self._set_slot(idx, 2, code, spin, origin, energize)
                return True
            self.pauli_blocks += 1
            return False
        # Бозон: свободный слот, затем стек
        if self.node_t[idx] < 0:
            self._set_slot(idx, 1, code, spin, origin, energize)
            return True
        if self.node_t2[idx] < 0:
            self._set_slot(idx, 2, code, spin, origin, energize)
            return True
        if self.stack_extra[idx] < BOSE_STACK_MAX:
            self.stack_extra[idx] += 1
            self.bose_stacks += 1
            if energize:
                self._led("bose", 0.1)
                self.energy[idx] += 0.1
                self.prob_real[idx] = 0.7
            self._sync_views()
            return True
        return False

    def _move_node(self, src: int, dst: int, slot: int = None) -> bool:
        """Переместить узел между ячейками (тип/спин/возраст сохраняются).
        При EM-динамике переносит непрерывную позицию/скорость и связанное
        состояние узла (psi-пин + энергия 1.2): ΣE сохраняется точно
        (перенос, а не источник). Гейт em_dynamics: старый сценарий
        (фазы 1–8, перемещения струнами) бит-идентичен.
        slot — конкретный слот узла (EM: в ячейке может быть по узлу в
        каждом слоте — без явного слота переехал бы «не тот» узел);
        None — старая логика (слот 1, если занят, иначе 2)."""
        if src == dst:
            return True
        if slot is None:
            slot = 1 if self.node_t[src] >= 0 else 2
        if slot == 1:
            if self.node_t[src] < 0:
                return False
            code, spin = int(self.node_t[src]), int(self.spin[src])
            age, org = int(self.matter_age[src]), int(self.node_origin[src])
            self.node_t[src] = -1
            self.spin[src] = -1
            self.matter_age[src] = 0
            self.node_origin[src] = -1
            self._set_slot(dst, 1, code, spin, org, energize=False, age=age)
            if self.em_dynamics:
                self._carry_node_state(src, dst, 1)
        else:
            if self.node_t2[src] < 0:
                return False
            code, spin = int(self.node_t2[src]), int(self.spin2[src])
            age, org = int(self.matter_age2[src]), int(self.node_origin2[src])
            self.node_t2[src] = -1
            self.spin2[src] = -1
            self.matter_age2[src] = 0
            self.node_origin2[src] = -1
            self._set_slot(dst, 2, code, spin, org, energize=False, age=age)
            if self.em_dynamics:
                self._carry_node_state(src, dst, 2)
        return True

    def _carry_node_state(self, src: int, dst: int, slot: int):
        """Перенос связанного состояния узла в новую ячейку (EM-динамика):
        энергия узла (1.2, ΣE-консервативно) и psi-пин (PSI_NODE_AMP).
        Без пина узел в новой ячейке теряет синтез (<0.15) и распадался бы.
        Волновая функция движется вместе с узлом: psi-шлейф в старой ячейке
        очищается — иначе след амплитуды 0.7 вызывает спонтанные рождения
        узлов на орбите, которые блокируют путь электрону.
        ВАЖНО: psi — на ячейку, а не на слот; чистим только когда ячейка
        стала полностью пустой (иначе убиваем синтез второго узла)."""
        if self.node_t[src] < 0 and self.node_t2[src] < 0:
            self.prob_real[src] = 0.0
            self.prob_imag[src] = 0.0
        if slot == 1:
            self.energy[dst] += float(self.energy[src]) - ENERGY_ZERO_POINT
            self.energy[src] = ENERGY_ZERO_POINT
            self.prob_real[dst] = PSI_NODE_AMP
            self.prob_imag[dst] = 0.0
            self.pos_x[dst] = self.pos_x[src]
            self.pos_y[dst] = self.pos_y[src]
            self.pos_z[dst] = self.pos_z[src]
            self.vel_x[dst] = self.vel_x[src]
            self.vel_y[dst] = self.vel_y[src]
            self.vel_z[dst] = self.vel_z[src]
        else:
            self.energy[dst] += float(self.energy[src]) - ENERGY_ZERO_POINT
            self.energy[src] = ENERGY_ZERO_POINT
            self.prob_real[dst] = PSI_NODE_AMP
            self.prob_imag[dst] = 0.0
            self.pos2_x[dst] = self.pos2_x[src]
            self.pos2_y[dst] = self.pos2_y[src]
            self.pos2_z[dst] = self.pos2_z[src]
            self.vel2_x[dst] = self.vel2_x[src]
            self.vel2_y[dst] = self.vel2_y[src]
            self.vel2_z[dst] = self.vel2_z[src]

    def _log_event(self, etype: str, pos, frm: str, to, parent=None, depth: int = 0) -> dict:
        """Событие динамики (распад/разрыв/аннигиляция) -> decay_events."""
        self._decay_id += 1
        ev = {"t": self.time_step, "id": self._decay_id,
              "parent": parent, "pos": [int(p) for p in pos],
              "from": frm, "to": to, "depth": depth, "type": etype}
        self.decay_events.append(ev)
        self._events_by_id[ev["id"]] = ev
        if len(self.decay_events) > 1000:
            del self.decay_events[:-500]
            self._events_by_id = {e["id"]: e for e in self.decay_events}
        return ev

    def _decay_node(self, idx: int, slot: int):
        """Распад узла: дочерний (цепочка τ→μ/π→e) или только энергия.
        Событие несёт parent/depth для реконструкции цепочек."""
        if slot == 1:
            code = int(self.node_t[idx])
            origin = int(self.node_origin[idx])
        else:
            code = int(self.node_t2[idx])
            origin = int(self.node_origin2[idx])
        if code < 0:
            return
        sym = _NODE_TYPES[code]
        if not TYPE_DECAY.get(sym):
            return  # фоновый распад (n): только энергия, без события
        # Канал распада (BR из TYPE_DECAY)
        to_sym = None
        channels = TYPE_DECAY.get(sym)
        if channels:
            r = float(self.rng.random())
            acc = 0.0
            for dsym, br in channels:
                acc += br
                if r < acc:
                    to_sym = dsym
                    break
        depth = 0
        if origin >= 0 and origin in self._events_by_id:
            depth = self._events_by_id[origin]["depth"] + 1
        x = idx // (self.size * self.size)
        y = (idx // self.size) % self.size
        z = idx % self.size
        ev = self._log_event("decay", [x, y, z], sym, to_sym,
                             parent=origin if origin >= 0 else None, depth=depth)
        if to_sym is not None:
            # Дочерний -> свободный слот в той же или соседней ячейке (тор)
            to_code = TYPE_CODE[to_sym]
            dspin = int(self.rng.integers(2))
            cells = [idx]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        cells.append(self._flat((x + dx) % self.size,
                                                (y + dy) % self.size,
                                                (z + dz) % self.size))
            placed = False
            for c in cells:
                if self._place_node(c, to_code, dspin, origin=ev["id"],
                                    energize=True):
                    placed = True
                    break
            if not placed:
                ev["to"] = None  # дочерний не разместился -> только энергия

    # --------------------------------------------------------
    # КВАРКОВЫЕ СТРУНЫ (конфайнмент: V = σ·d)
    # --------------------------------------------------------

    def _cell_is_quark(self, idx: int) -> bool:
        # все 4 флейвора (в основном мире спонтанно рождаются только u/ū)
        quarks = (TYPE_CODE["u"], TYPE_CODE[_ANTI_U],
                  TYPE_CODE["d"], TYPE_CODE[_ANTI_D])
        return (int(self.node_t[idx]) in quarks) or (int(self.node_t2[idx]) in quarks)

    @staticmethod
    def _tor_diff(a: int, b: int, s: int) -> int:
        """Знаковое торическое расстояние b - a (кратчайший путь)."""
        d = (b - a) % s
        if d > s // 2:
            d -= s
        return d

    def _update_strings(self):
        """Струны: d ≥ STRING_BREAK -> разрыв (пара q q̄ в середине, 2 новые);
        d ≤ 1 -> u + ū аннигилируют в энергию (как e⁺e⁻ → 2γ);
        иначе стягивание (P=0.5): конец сдвигается к другому на 1 ячейку
        вдоль оси максимального |Δ| (если цель свободна)."""
        if not self.strings:
            return
        s = self.size
        keep = []
        new_strs = []
        for st in self.strings:
            a, b = st["a"], st["b"]
            if not (self._cell_is_quark(a) and self._cell_is_quark(b)):
                continue  # конец исчез -> струна рассеивается
            ax, ay, az = a // (s * s), (a // s) % s, a % s
            bx, by, bz = b // (s * s), (b // s) % s, b % s
            dx = self._tor_diff(ax, bx, s)
            dy = self._tor_diff(ay, by, s)
            dz = self._tor_diff(az, bz, s)
            d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if d >= STRING_BREAK:
                self._break_string(st, (dx, dy, dz), d, new_strs)
                continue
            if d <= 1.0 + 1e-9:
                # u + ū встретились -> аннигиляция в энергию
                self.node_t[a] = -1
                self.node_t2[a] = -1
                self.spin[a] = -1
                self.spin2[a] = -1
                self.stack_extra[a] = 0
                self._led("annihilation", 0.2)
                self.energy[a] += 0.2
                self._log_event("annihilation", [ax, ay, az], "u+ū", None)
                self._sync_views()
                continue
            # Стягивание (P=0.5): сдвинуть конец a к b
            if float(self.rng.random()) < 0.5:
                ax_, ay_, az_ = abs(dx), abs(dy), abs(dz)
                if ax_ >= ay_ and ax_ >= az_:
                    npos = (ax + (1 if dx > 0 else -1), ay, az)
                elif ay_ >= az_:
                    npos = (ax, ay + (1 if dy > 0 else -1), az)
                else:
                    npos = (ax, ay, az + (1 if dz > 0 else -1))
                n_idx = self._flat(*[p % s for p in npos])
                if not (self.node_t[n_idx] >= 0 or self.node_t2[n_idx] >= 0
                        or self.stack_extra[n_idx] > 0):
                    if self._move_node(a, n_idx):
                        st["a"] = n_idx
            keep.append(st)
        self.strings = keep + new_strs

    def _break_string(self, st: dict, delta: tuple, d: float, new_strs: list):
        """Разрыв струны: пара q q̄ в середине, 2 новые струны, энергия σ·d/2
        на каждый продукт (депозиция в ячейки)."""
        s = self.size
        a, b = st["a"], st["b"]
        ax, ay, az = a // (s * s), (a // s) % s, a % s
        m = [(ax + int(round(delta[0] / 2))) % s,
             (ay + int(round(delta[1] / 2))) % s,
             (az + int(round(delta[2] / 2))) % s]
        m1 = self._flat(*m)
        m2 = self._flat((m[0] + 1) % s, m[1], m[2])
        c_u, c_uu = TYPE_CODE["u"], TYPE_CODE[_ANTI_U]
        ok1 = self._place_node(m1, c_u, 1, energize=True)
        ok2 = self._place_node(m2, c_uu, 0, energize=True)
        self._log_event("string_break", list(m), "string",
                        "u+ū" if (ok1 and ok2) else None)
        if not (ok1 and ok2):
            # Нет свободных ячеек: энергия струны -> тепло у конца
            self._led("string_break", STRING_TENSION * d)
            self.energy[a] += STRING_TENSION * d
            self._sync_views()
            return
        self._led("string_break", STRING_TENSION * d)
        self.energy[m1] += STRING_TENSION * d / 2
        self.energy[m2] += STRING_TENSION * d / 2
        new_strs.append({"a": a, "b": m1, "born": self.time_step})
        new_strs.append({"a": b, "b": m2, "born": self.time_step})
        self.n_strings_broken += 1
        self.n_quarks_created += 2
        self._sync_views()

    # --------------------------------------------------------
    # ХАДРОНИЗАЦИЯ (фазы 9–11): цветовой синглет uud→p, udd→n
    # --------------------------------------------------------

    @staticmethod
    def _cell_coords(idx: int, s: int):
        return (idx // (s * s), (idx // s) % s, idx % s)

    @staticmethod
    def _tor_dist3(a, b, s: int) -> float:
        """Расстояние между координатами ячеек на торе."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dz = abs(a[2] - b[2])
        dx = min(dx, s - dx)
        dy = min(dy, s - dy)
        dz = min(dz, s - dz)
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    def _quark_slots(self):
        """Занятые кварковые слоты: (slot, idx, sym)."""
        qsyms = {"u", _ANTI_U, "d", _ANTI_D}
        out = []
        for slot, tarr in ((1, self.node_t), (2, self.node_t2)):
            for i in np.where(tarr >= 0)[0]:
                sym = _NODE_TYPES[int(tarr[i])]
                if sym in qsyms:
                    out.append((slot, int(i), sym))
        return out

    def _clear_slot(self, slot: int, idx: int):
        """Очистить слот (тип/спин/возраст/origin/скорость); pos = ячейка."""
        s = self.size
        if slot == 1:
            self.node_t[idx] = -1
            self.spin[idx] = -1
            self.matter_age[idx] = 0
            self.node_origin[idx] = -1
        else:
            self.node_t2[idx] = -1
            self.spin2[idx] = -1
            self.matter_age2[idx] = 0
            self.node_origin2[idx] = -1
        cx, cy, cz = idx // (s * s), (idx // s) % s, idx % s
        if slot == 1:
            self.pos_x[idx], self.pos_y[idx], self.pos_z[idx] = cx, cy, cz
            self.vel_x[idx] = 0.0
            self.vel_y[idx] = 0.0
            self.vel_z[idx] = 0.0
        else:
            self.pos2_x[idx], self.pos2_y[idx], self.pos2_z[idx] = cx, cy, cz
            self.vel2_x[idx] = 0.0
            self.vel2_y[idx] = 0.0
            self.vel2_z[idx] = 0.0
        self._sync_views()

    def hadronize(self) -> bool:
        """Хадронизация: цветовой синглет в радиусе ≤ HADRON_RADIUS сливается
        в адрон: {u,u,d}→p, {u,d,d}→n. Массовый избыток (m_h − Σm_q, PDG)
        депонируется как энергия поля в ячейке слияния (ledger 'hadronization').
        True — произошла хотя бы одна реакция. Вызывается явно (не в step())."""
        quarks = self._quark_slots()
        s = self.size
        for a in range(len(quarks)):
            for b in range(a + 1, len(quarks)):
                for c in range(b + 1, len(quarks)):
                    qa, qb, qc = quarks[a], quarks[b], quarks[c]
                    pa = self._cell_coords(qa[1], s)
                    pb = self._cell_coords(qb[1], s)
                    pc = self._cell_coords(qc[1], s)
                    if (self._tor_dist3(pa, pb, s) > HADRON_RADIUS
                            or self._tor_dist3(pa, pc, s) > HADRON_RADIUS
                            or self._tor_dist3(pb, pc, s) > HADRON_RADIUS):
                        continue
                    combo = "".join(sorted(x[2] for x in (qa, qb, qc)))
                    had = {"duu": "p", "ddu": "n"}.get(combo)   # отсортированный состав
                    if had is None:   # uuu, uūd, ūū... — не цветовой синглет
                        continue
                    # Центр слияния: центроид (ячейки кварков освобождаются ниже,
                    # поэтому занятость проверяется с их исключением)
                    ctr = (int(round((pa[0] + pb[0] + pc[0]) / 3.0)) % s,
                           int(round((pa[1] + pb[1] + pc[1]) / 3.0)) % s,
                           int(round((pa[2] + pb[2] + pc[2]) / 3.0)) % s)
                    qcells = {qa[1], qb[1], qc[1]}

                    def _free(i: int) -> bool:
                        return (i in qcells
                                or (self.node_t[i] < 0 and self.node_t2[i] < 0))

                    target = self._flat(*ctr)
                    if not _free(target):
                        placed = None
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                for dz in (-1, 0, 1):
                                    if dx == dy == dz == 0:
                                        continue
                                    nidx = self._flat(
                                        (ctr[0] + dx) % s, (ctr[1] + dy) % s,
                                        (ctr[2] + dz) % s)
                                    if _free(nidx):
                                        placed = nidx
                                        break
                                if placed is not None:
                                    break
                            if placed is not None:
                                break
                        if placed is None:
                            continue
                        target = placed
                    # Событие ДО очистки слотов (origin для адрона)
                    ev = self._log_event("hadronization", list(ctr), combo, had)
                    for slot, i, _sym in (qa, qb, qc):
                        self._clear_slot(slot, i)
                    # Слоты цели свободны (проверено выше) → размещение точно
                    spin = int(self.rng.integers(2))
                    self._place_node(target, TYPE_CODE[had], spin,
                                     origin=ev["id"], energize=True)
                    self._led("hadronization", HADRON_RELEASE[had])
                    self.energy[target] += HADRON_RELEASE[had]
                    self.n_hadronized += 1
                    return True
        return False

    def _psi_abs_now(self):
        return np.sqrt(self.prob_real ** 2 + self.prob_imag ** 2)

    def _field_mag_now(self):
        return np.sqrt(np.sum(self.field ** 2, axis=0)).flatten()

    # --------------------------------------------------------
    # ЭМ-ДИНАМИКА (фазы 10–11): кулон заряженных узлов
    # --------------------------------------------------------

    def _update_em(self):
        """Кулоновское взаимодействие заряженных узлов (гейт em_dynamics).
        F = K·q₁q₂·r/(d² + ε²)^{3/2} (тор, сглаживание Платтера, консервативно);
        симплициев Евлер:
        v += F/m_eff·dt, pos += v·dt (тор, |v| ≤ EM_VMAX).
        Пересечение ячейки: round(pos) → _move_node; цель занята тем же
        спин-слотом → упругий откат (pos = ячейка, v *= 0.5).
        Детерминированно, БЕЗ RNG; слой L1 (энергия) не затрагивается."""
        if not self.em_dynamics:
            return
        s = self.size
        # Заряженные узлы: (slot, idx, charge, m_eff)
        occ = []
        for slot, tarr in ((1, self.node_t), (2, self.node_t2)):
            for i in np.where(tarr >= 0)[0]:
                sym = _NODE_TYPES[int(tarr[i])]
                q = _NODE_CHARGE[sym]
                m = self._mef[sym]
                if q == 0.0 or m <= 0.0:
                    continue
                occ.append((slot, int(i), q, m))
        n = len(occ)
        if n < 2:
            return
        P = np.empty((n, 3))
        V = np.empty((n, 3))
        for k, (slot, i, q, m) in enumerate(occ):
            if slot == 1:
                P[k] = (self.pos_x[i], self.pos_y[i], self.pos_z[i])
                V[k] = (self.vel_x[i], self.vel_y[i], self.vel_z[i])
            else:
                P[k] = (self.pos2_x[i], self.pos2_y[i], self.pos2_z[i])
                V[k] = (self.vel2_x[i], self.vel2_y[i], self.vel2_z[i])
        # Кулоновские силы O(n²), векторизовано чанками по строкам (n² при
        # n=10³ — миллисекунды; память чанка ≤ ~60 МБ). D[i,j] — кратчайший
        # вектор P[i]−P[j] на торе; консервативная сила (Платтер):
        # F[i] = Σ_j K·q_i·q_j·D[i,j]/(D2[i,j]+ε²)^{3/2}, диагональ = 0.
        # (Порядок суммирования отличается от старого O(n²) цикла на уровне
        # округления float64; физика идентична.)
        F = np.zeros((n, 3))
        q = np.array([o[2] for o in occ], dtype=np.float64)
        m = np.array([o[3] for o in occ], dtype=np.float64)
        half = 0.5 * s
        chunk = max(16, min(512, 2_500_000 // n))
        for a0 in range(0, n, chunk):
            a1 = min(a0 + chunk, n)
            D = P[a0:a1, None, :] - P[None, :, :]
            D = np.mod(D + half, s) - half
            D2 = np.einsum("ijk,ijk->ij", D, D)
            W = (COULOMB_K * q[a0:a1, None] * q[None, :]
                 / (D2 + COULOMB_SOFT2) ** 1.5)
            W[np.arange(a1 - a0), np.arange(a0, a1)] = 0.0
            F[a0:a1] = np.einsum("ij,ijk->ik", W, D)
        # Симплициев Евлер (векторизованно): v += F/m·dt → pos += v·dt
        V += F / m[:, None]
        sp = np.linalg.norm(V, axis=1)
        over = sp > EM_VMAX
        V[over] *= (EM_VMAX / sp[over])[:, None]
        P = np.mod(P + V * EM_DT, s)
        # Запись назад + пересечение ячеек
        for k, (slot, i, q, m) in enumerate(occ):
            if slot == 1:
                self.pos_x[i], self.pos_y[i], self.pos_z[i] = P[k]
                self.vel_x[i], self.vel_y[i], self.vel_z[i] = V[k]
            else:
                self.pos2_x[i], self.pos2_y[i], self.pos2_z[i] = P[k]
                self.vel2_x[i], self.vel2_y[i], self.vel2_z[i] = V[k]
            px, py, pz = (int(round(c)) % s for c in P[k])
            nidx = self._flat(px, py, pz)
            if nidx == i:
                continue
            free = (self.node_t[nidx] < 0) if slot == 1 else (self.node_t2[nidx] < 0)
            if free:
                self._move_node(i, nidx, slot)   # переносит pos/vel + состояние узла
            else:
                # Упругий откат (без RNG): pos = ячейка, v *= 0.5
                cx, cy, cz = i // (s * s), (i // s) % s, i % s
                if slot == 1:
                    self.pos_x[i], self.pos_y[i], self.pos_z[i] = cx, cy, cz
                    self.vel_x[i] *= 0.5
                    self.vel_y[i] *= 0.5
                    self.vel_z[i] *= 0.5
                else:
                    self.pos2_x[i], self.pos2_y[i], self.pos2_z[i] = cx, cy, cz
                    self.vel2_x[i] *= 0.5
                    self.vel2_y[i] *= 0.5
                    self.vel2_z[i] *= 0.5

    # --------------------------------------------------------
    # ЭНТРОПИЙНЫЙ РЕГУЛЯТОР
    # --------------------------------------------------------

    def _entropy_regulator(self):
        """
        Мониторит энтропию и применяет давление:
        - S > TARGET: «охлаждение» (порядок, синтез)
        - S < TARGET: «нагрев» (хаос, раслоение)
        Аналог: термостат / тёмная энергия.
        """
        self._entropy_counter += 1
        if self._entropy_counter < ENTROPY_RESPONSE_TIME:
            return
        self._entropy_counter = 0

        if not self.entropy_regulator_active:
            self.entropy_pressure_current = 0.0
            return

        deviation = self.global_entropy - ENTROPY_TARGET
        self.entropy_pressure_current = float(np.clip(deviation, -1.0, 1.0) * ENTROPY_PRESSURE)
        P = self.entropy_pressure_current

        if P > 0:
            # Слишком много хаоса -> «охлаждение»
            self.prob_real *= (1.0 - P)
            self.prob_imag *= (1.0 - P)
            self._led("reg_cool", float(P * 0.5 * self.energy.sum()))
            self.energy *= (1.0 + P * 0.5)
            self.field *= (1.0 - P * 0.3)
        elif P < 0:
            # Слишком много порядка -> «нагрев»
            a = abs(P)
            self.prob_real += self.rng.normal(0, a * 0.1, self.n)
            self.prob_imag += self.rng.normal(0, a * 0.1, self.n)
            self._led("reg_heat", a * 0.01 * self.energy.size)
            self.energy += a * 0.01
            self.field += self.rng.normal(0, a * 0.05, self.field.shape)

    # --------------------------------------------------------
    # ГЛОБАЛЬНЫЕ МЕТРИКИ
    # --------------------------------------------------------

    def _update_metrics(self):
        states = np.stack([
            np.sin(self.time_phase),
            self.energy,
            self._psi_abs_now(),
            -self.grav,
            self._field_mag_now(),
        ])
        self.entropy_local = states.std(axis=0)
        self.global_entropy = float(self.entropy_local.mean())
        self.total_energy = float(self.energy.sum())
        self.matter_count = int(self.stack.sum())

        self.entropy_history.append(self.global_entropy)
        self.energy_history.append(self.total_energy)
        self.matter_history.append(self.matter_count)

        stats = {}
        for layer in range(N_LAYERS):
            v = self._layer_values(layer)
            stats[LAYER_NAMES[layer]] = {
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
                "max": round(float(np.abs(v).max()), 4),
            }
        self.layer_stats_history.append(stats)

        self._recompute_derived()

    # --------------------------------------------------------
    # ВНЕШНИЕ ВОЗДЕЙСТВИЯ
    # --------------------------------------------------------

    def _apply_injection(self, injection: dict):
        """Впрыск энергии/заряда в точку (с затуханием 1/r^2)."""
        x, y, z = injection["pos"]
        radius = injection.get("radius", 2)
        energy = injection.get("energy", 0.5)
        charge = injection.get("charge", 0.0)

        tot = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                        dist2 = dx * dx + dy * dy + dz * dz
                        falloff = 1.0 / (1.0 + dist2 * 0.5)
                        idx = self._flat(nx, ny, nz)
                        self.energy[idx] += energy * falloff
                        self.charge[idx] += charge * falloff
                        self.prob_real[idx] += energy * falloff * 0.3
                        tot += energy * falloff
        self._led("injection", tot)

    def inject_particle(self, pos, symbol="n", spin=None, origin: int = -1) -> bool:
        """Создать частицу (узел) в точке. symbol — из каталога (по умолчанию "n").
        spin: 0 (↓) или 1 (↑); None — случайный. origin — id события создания
        (для отслеживания узла). False = Pauli-блок."""
        code = TYPE_CODE.get(symbol, 0)
        if spin is None:
            spin = int(self.rng.integers(2))
        x, y, z = pos
        idx = self._flat(x, y, z)
        ok = self._place_node(idx, code, int(spin), origin=origin, energize=True)
        if ok and _base_symbol(symbol) in ("u", "d"):
            self.n_quarks_created += 1
        return ok

    def inject_quark_pair(self, pos, dist: int = 2, axis: int = 0,
                          flavor: str = "u") -> bool:
        """Кварковая пара q — q̄ со струной между (конфайнмент).
        flavor: 'u' или 'd' (по умолчанию 'u' — как в старом сценарии)."""
        s = self.size
        p2 = [pos[0], pos[1], pos[2]]
        p2[axis] = (p2[axis] + dist) % s
        a = self._flat(*pos)
        b = self._flat(*p2)
        c_q = TYPE_CODE["u"] if flavor == "u" else TYPE_CODE["d"]
        c_aq = TYPE_CODE[_ANTI_U] if flavor == "u" else TYPE_CODE[_ANTI_D]
        ok1 = self._place_node(a, c_q, 1, energize=True)
        ok2 = self._place_node(b, c_aq, 0, energize=True)
        if ok1 and ok2:
            self.strings.append({"a": a, "b": b, "born": self.time_step})
            self.n_quarks_created += 2
            self._sync_views()
        return bool(ok1 and ok2)

    def apply_force(self, pos, magnitude: float = 0.5):
        """Применить силу (аналог «ломаем дерево»: деформация + нагрев)."""
        x, y, z = pos
        idxs = []
        falloffs = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist < 3:
                            idxs.append(self._flat(nx, ny, nz))
                            falloffs.append(1.0 / (1.0 + dist))
        idxs = np.array(idxs)
        falloffs = np.array(falloffs)

        de = self.rng.uniform(-magnitude, magnitude, len(idxs)) * falloffs
        self.energy[idxs] += de
        self._led("force", float(de.sum()))
        self.prob_real[idxs] += self.rng.uniform(-magnitude, magnitude, len(idxs)) * falloffs * 0.5
        self.prob_imag[idxs] += self.rng.uniform(-magnitude, magnitude, len(idxs)) * falloffs * 0.25
        f = self.field.reshape(3, -1)  # view (3, n)
        f[:, idxs] += self.rng.normal(0, magnitude * 0.3, (3, len(idxs))) * falloffs[None, :]
        self.grav[idxs] -= magnitude * falloffs * 0.1  # деформация

    # --------------------------------------------------------
    # ТРЕКИНГ УЗЛОВ И ГЕОМЕТРИЯ ТОРА (фазы 10–11)
    # --------------------------------------------------------

    def _slot_pos(self, slot: int, idx: int):
        if slot == 1:
            return (float(self.pos_x[idx]), float(self.pos_y[idx]),
                    float(self.pos_z[idx]))
        return (float(self.pos2_x[idx]), float(self.pos2_y[idx]),
                float(self.pos2_z[idx]))

    def find_node(self, origin_id: int):
        """Узел по id события создания: {slot, idx, code, pos} или None."""
        for slot, tarr, oarr in ((1, self.node_t, self.node_origin),
                                 (2, self.node_t2, self.node_origin2)):
            hit = np.where((oarr == origin_id) & (tarr >= 0))[0]
            if len(hit):
                i = int(hit[0])
                return {"slot": slot, "idx": i, "code": int(tarr[i]),
                        "pos": self._slot_pos(slot, i)}
        return None

    def torus_distance(self, pa, pb) -> float:
        """Расстояние между двумя точками (float-координаты) на торе."""
        s = self.size
        d = np.asarray(pb, dtype=np.float64) - np.asarray(pa, dtype=np.float64)
        d = np.mod(d + 0.5 * s, s) - 0.5 * s
        return float(math.sqrt(float(d @ d)))

    def node_distance(self, id_a: int, id_b: int) -> float:
        """Расстояние между двумя узлами по id создания (узел живёт)."""
        na, nb = self.find_node(id_a), self.find_node(id_b)
        if na is None or nb is None:
            raise RuntimeError("узел исчез (origin id не найден)")
        return self.torus_distance(na["pos"], nb["pos"])

    # --------------------------------------------------------
    # АТОМЫ И МОЛЕКУЛЫ (фазы 10–11)
    # --------------------------------------------------------

    def _orbital_velocity(self, r: float, q_a: float = 1.0, q_b: float = 1.0,
                          m_sym: str = "e⁻") -> float:
        """Скорость круговой кулоновской орбиты (сглаживание Платтера):
        m·v²/r = K·|q₁q₂|·r/(r²+ε²)^{3/2}  ⇒  v² = K·|q₁q₂|·r²/(m_eff·(r²+ε²)^{3/2}).
        Масса — из self._mef (legacy: _NODE_MEFF; unified: _NODE_MEFF_UNI, F1)."""
        r = max(float(r), 0.1)
        m = self._mef[m_sym]
        return math.sqrt(COULOMB_K * abs(q_a * q_b) * r ** 2 /
                         (m * (r ** 2 + COULOMB_SOFT2) ** 1.5))

    def _set_node_velocity(self, node: dict, axis: int, value: float):
        """Задать скорость узла (по slot/idx из find_node) по оси 0..2."""
        slot, i = node["slot"], node["idx"]
        if slot == 1:
            for arr, ax in ((self.vel_x, 0), (self.vel_y, 1), (self.vel_z, 2)):
                if ax == axis:
                    arr[i] = value
        else:
            for arr, ax in ((self.vel2_x, 0), (self.vel2_y, 1), (self.vel2_z, 2)):
                if ax == axis:
                    arr[i] = value

    def inject_hydrogen(self, center, r0: int = None) -> dict:
        """Атом водорода: p в центре, e⁻ на расстоянии r0 (ячейки) по +X с
        тангенциальной скоростью круговой орбиты (+Y).
        Возвращает {'p', 'e' (id событий), 'r0'}. Для орбиты включите
        em_dynamics ДО вызова."""
        if r0 is None:
            r0 = max(2, int(round(self.size / 7)))
        s = self.size
        c = tuple(int(v) % s for v in center)
        ev_p = self._log_event("creation", list(c), "p", None)
        if not self.inject_particle(c, "p", spin=0, origin=ev_p["id"]):
            raise ValueError(f"ячейка {c} занята (Pauli-блок)")
        epos = ((c[0] + r0) % s, c[1], c[2])
        ev_e = self._log_event("creation", list(epos), "e⁻", None)
        if not self.inject_particle(epos, "e⁻", spin=1, origin=ev_e["id"]):
            raise ValueError(f"ячейка {epos} занята (Pauli-блок)")
        self._set_node_velocity(self.find_node(ev_e["id"]), 1,
                                self._orbital_velocity(r0))
        return {"p": ev_p["id"], "e": ev_e["id"], "r0": float(r0)}

    def inject_h2(self, center, d0: int = None, r0: int = None) -> dict:
        """Молекула H2: 2 протона по +X (длина связи d0), оба электрона в
        середине связи (общая «связывающая орбиталь»: одна ячейка mid,
        противоположные спины → Pauli соблюдён), v=0. Электроны удерживаются
        в глубоком потенциальном колодце между протонами — связь стабильна.
        r0 — справочный радиус для проверки удержания электронов.
        Возвращает {'p1','p2','e1','e2' (id), 'd0','r0'}. Для динамики
        включите em_dynamics ДО вызова. d0 ≥ 2 (нужна ячейка между ядрами)."""
        if d0 is None:
            d0 = max(2, int(round(self.size / 10)))
        if r0 is None:
            r0 = max(2, int(round(self.size / 7)))
        s = self.size
        c = tuple(int(v) % s for v in center)
        p1 = c
        p2 = ((c[0] + d0) % s, c[1], c[2])
        mid = ((c[0] + d0 // 2) % s, c[1], c[2])
        out = {}
        for key, pos, sym, spin in (("p1", p1, "p", 0), ("p2", p2, "p", 1),
                                    ("e1", mid, "e⁻", 0), ("e2", mid, "e⁻", 1)):
            ev = self._log_event("creation", list(pos), sym, None)
            if not self.inject_particle(pos, sym, spin=spin, origin=ev["id"]):
                raise ValueError(f"ячейка {pos} занята (Pauli-блок)")
            out[key] = ev["id"]
        out["d0"], out["r0"] = float(d0), float(r0)
        return out

    def find_free_center(self, desired, offsets=((0, 0, 0),)):
        """Детерминированный поиск свободного центра: все ячейки
        {center + off} пусты (оба спин-слота). Полный обход тора в фикс.
        порядке от desired (без RNG). None, если свободного места нет."""
        s = self.size
        d = tuple(int(v) % s for v in desired)
        offs = [tuple(int(o) % s for o in o3) for o3 in offsets]

        def free(center):
            for o in offs:
                idx = self._flat(*[ (center[k] + o[k]) % s for k in range(3) ])
                if self.node_t[idx] >= 0 or self.node_t2[idx] >= 0:
                    return False
            return True

        for dx in range(s):
            for dy in range(s):
                for dz in range(s):
                    cand = ((d[0] + dx) % s, (d[1] + dy) % s, (d[2] + dz) % s)
                    if free(cand):
                        return cand
        return None

    # --------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ (терминал)
    # --------------------------------------------------------

    def visualize_layers(self) -> str:
        lines = []
        lines.append("\n" + "─" * 60)
        lines.append(f"  СЛОЙ                     СР.ЗНАЧ  ДИСПЕРСИЯ  МАКС  СТАТУС")
        lines.append(f"  {'─' * 56}")
        for layer in range(N_LAYERS):
            v = self._layer_values(layer)
            status = "● АКТИВЕН" if v.std() > 0.05 else "○ СПИТ"
            lines.append(
                f"  {LAYER_NAMES[layer]:<24} "
                f"{v.mean():<10.4f} "
                f"{v.std():<10.4f} "
                f"{np.abs(v).max():<6.4f} "
                f"{status}"
            )
        lines.append(f"  {'─' * 56}")
        lines.append(
            f"  {'ЭНТРОПИЯ (глобальная)':<24} "
            f"{self.global_entropy:<10.4f} {'':<6} {'ЦЕЛЬ=' + str(ENTROPY_TARGET)}"
        )
        lines.append(f"  {'ДАВЛЕНИЕ РЕГУЛЯТОРА':<24} {self.entropy_pressure_current:<10.4f}")
        lines.append(f"  {'МАТЕРИЯ (узлов)':<24} {self.matter_count:<10d}")
        lines.append("─" * 60)
        return "\n".join(lines)

    def _cell_char(self, idx: int) -> str:
        """Ячейка -> символ ASCII."""
        if self.matter[idx]:
            return "★"
        e = self.energy[idx]
        fm = self.field_mag[idx]
        p = self.psi_abs[idx]
        if e > 0.5:
            return "●"
        elif fm > 0.3:
            return "▲"
        elif p > 0.4:
            return "◆"
        elif e > 0.15:
            return "○"
        elif fm > 0.1:
            return "▼"
        return "·"

    def visualize_slice(self, axis: int = 1, index: int = None) -> str:
        """
        Визуализация 3D среза (ASCII).
        axis: 0=X-срез, 1=Y-срез, 2=Z-срез; index: позиция среза.
        """
        if index is None:
            index = self.size // 2
        s = self.size

        if axis == 0:    # X фиксирован: строки=y, столбцы=z
            row_ax, col_ax = 1, 2
            plane_name = f"X-срез (X={index})"
        elif axis == 1:  # Y фиксирован: строки=x, столбцы=z
            row_ax, col_ax = 0, 2
            plane_name = f"Y-срез (Y={index})"
        else:            # Z фиксирован: строки=x, столбцы=y
            row_ax, col_ax = 0, 1
            plane_name = f"Z-срез (Z={index})"

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append(f"  СРЕЗ: {plane_name}")
        lines.append(f"  Шаг: {self.time_step} | Энтропия: {self.global_entropy:.4f} | "
                     f"Энергия: {self.total_energy:.2f} | Материя: {self.matter_count}")
        lines.append(f"  Регулятор: P={self.entropy_pressure_current:+.4f}")
        lines.append("=" * 60)
        lines.append("\n  Значки:  ★=материя  ●=выс.энергия  ○=низк.энергия")
        lines.append("           ▲=сильное поле  ▼=слабое поле  ◆=выс.вероятность  ·=вакуум")

        header = "  {:>4}".format("") + "".join(f"{j:>2}" for j in range(s))
        lines.append(header)
        for i in range(s):
            row = f"  {i:>4}"
            for j in range(s):
                coords = [0, 0, 0]
                coords[row_ax] = i
                coords[col_ax] = j
                coords[axis] = index
                idx = self._flat(coords[0], coords[1], coords[2])
                row += f"{self._cell_char(idx):>2}"
            lines.append(row)
        return "\n".join(lines)

    def visualize_3d_preview(self, n_slices: int = 3) -> str:
        """3D превью: несколько Y-срезов (каждый 2-й пиксель)."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("  3D ПРЕВЬЮ (срезы по Y-оси)")
        lines.append("=" * 60)
        s = self.size
        for idx in np.linspace(2, s - 3, n_slices, dtype=int):
            lines.append(f"\n  ─── Y = {idx} ───")
            for y in range(0, s, 2):
                row = f"  {y:>3}|"
                for x in range(0, s, 2):
                    idx_c = self._flat(x, idx, y)
                    row += f" {self._cell_char(idx_c)}"
                lines.append(row)
        return "\n".join(lines)

    # --------------------------------------------------------
    # СНАПШОТ ДЛЯ 3D-ВЬЮВЕРА
    # --------------------------------------------------------

    def _metrics(self) -> dict:
        return {
            "entropy": round(self.global_entropy, 4),
            "target": ENTROPY_TARGET,
            "energy": round(self.total_energy, 2),
            "matter": self.matter_count,
            "strings": len(self.strings),
            "pauli": self.pauli_blocks,
            "decays": len(self.decay_events),
            "pressure": round(self.entropy_pressure_current, 4),
            "regulator": self.entropy_regulator_active,
            "em": self.em_dynamics,
            "hadronized": self.n_hadronized,
        }

    def _layer_stats(self) -> dict:
        layers = {}
        for layer in range(N_LAYERS):
            v = self._layer_values(layer)
            layers[LAYER_NAMES[layer]] = {
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
                "max": round(float(np.abs(v).max()), 4),
            }
        return layers

    def _strings_list(self) -> list:
        return [
            [st["a"] // (self.size * self.size),
             (st["a"] // self.size) % self.size, st["a"] % self.size,
             st["b"] // (self.size * self.size),
             (st["b"] // self.size) % self.size, st["b"] % self.size]
            for st in self.strings
        ]

    def _nodes_list(self) -> list:
        """Точный список узлов [[x, y, z, type_code, spin], …]: суб-ячеечные
        координаты (3 знака), код типа и спин; по 1–2 записи на занятую ячейку
        (спин-слоты) + стек бозонов. Мелкий (узлов << ячеек)."""
        nodes = []
        for tarr, px, py, pz, sp_arr in (
                (self.node_t, self.pos_x, self.pos_y, self.pos_z, self.spin),
                (self.node_t2, self.pos2_x, self.pos2_y, self.pos2_z, self.spin2)):
            for i in np.where(tarr >= 0)[0]:
                nodes.append([round(float(px[i]), 3),
                              round(float(py[i]), 3),
                              round(float(pz[i]), 3),
                              int(tarr[i]), int(sp_arr[i])])
        return nodes

    def snapshot(self, label: str) -> dict:
        """Кадр состояния объёма для интерактивной 3D-визуализации.
        data["nodes"] — точный список узлов (overlay поверх даунсемпл-фона)."""
        self._recompute_derived()
        return {
            "t": self.time_step,
            "label": label,
            "metrics": self._metrics(),
            "layers": self._layer_stats(),
            "data": {
                "time_phase": self.time_phase.round(4).tolist(),
                "time_rate": self.time_rate.round(4).tolist(),
                "energy": self.energy.round(4).tolist(),
                "psi_abs": self.psi_abs.round(4).tolist(),
                "psi_phase": self.psi_phase.round(4).tolist(),
                "grav": self.grav.round(4).tolist(),
                "field_mag": self.field_mag.round(4).tolist(),
                "field_vec": self.field.flatten().round(4).tolist(),  # [E_x(8000), E_y(8000), E_z(8000)]
                "matter": self.matter.astype(int).tolist(),
                "spin": np.where(self.node_t >= 0, self.spin,
                                 np.where(self.node_t2 >= 0, self.spin2, -1)).tolist(),
                "stack": self.stack.tolist(),
                "nodes": self._nodes_list(),
                "strings": self._strings_list(),
                "decays": self.decay_events[-100:],
            },
        }

    def snapshot_arrays(self, label: str, bg_res: int = None) -> dict:
        """Как snapshot(), но data — numpy-массивы (без tolist: list'ы из 10⁶
        чисел — сотни МБ памяти на кадр). bg_res — даунсемпл фона до bg_res³
        (mean для полей, max для matter/stack; spin по max, -1 сохраняется).
        Узлы/струны — точные (полное разрешение)."""
        self._recompute_derived()

        def f3(arr):
            a = arr.reshape(self.shape)
            if bg_res:
                a = self._downsample(a, bg_res)
            return a

        def f3max(arr):
            a = arr.reshape(self.shape).astype(np.int8)
            if bg_res:
                a = self._downsample(a, bg_res, mode="max")
            return a

        spin = np.where(self.node_t >= 0, self.spin,
                        np.where(self.node_t2 >= 0, self.spin2, -1))
        fv = self.field
        if bg_res:
            fv = self._downsample(fv, bg_res)
        return {
            "t": self.time_step,
            "label": label,
            "metrics": self._metrics(),
            "layers": self._layer_stats(),
            "data": {
                "time_phase": f3(self.time_phase),
                "time_rate": f3(self.time_rate),
                "energy": f3(self.energy),
                "psi_abs": f3(self.psi_abs),
                "psi_phase": f3(self.psi_phase),
                "grav": f3(self.grav),
                "field_mag": f3(self.field_mag),
                "field_vec": fv,  # (3, D, D, D)
                "matter": f3max(self.matter),
                "spin": f3max(spin),
                "stack": f3max(self.stack),
                "nodes": self._nodes_list(),
                "strings": self._strings_list(),
                "decays": self.decay_events[-100:],
            },
        }

    def _downsample(self, arr, D: int, mode: str = "mean"):
        """Даунсемпл сетки size³ до D³ (D должен делить size)."""
        return downsample(arr, self.size, D, mode=mode)

    def snapshot_nodes(self, label: str) -> dict:
        """Лёгкий кадр для анимации: только узлы (суб-ячеечные координаты,
        тип, спин) + струны — без полных полевых массивов (~1 КБ на кадр).
        Вьювер рисует тёмный фон и цветные точки узлов."""
        self._recompute_derived()
        return {
            "t": self.time_step,
            "label": label,
            "light": True,
            "metrics": self._metrics(),
            "layers": self._layer_stats(),
            "data": {
                "nodes": self._nodes_list(),
                "strings": self._strings_list(),
                "decays": self.decay_events[-100:],
            },
        }


# ============================================================
# СЦЕНАРИЙ ДЕМОНСТРАЦИИ
# ============================================================

def _banner():
    print("┌" + "─" * 62 + "┐")
    print("│" + "  ЦИФРОВАЯ МОДЕЛЬ 5 СУБПРОСТРАНСТВ В 3D ОБЪЁМЕ".center(58) + "│")
    print("│" + "  ВРЕМЯ · ЭНЕРГИЯ · ВЕРОЯТНОСТЬ · ГРАВИТАЦИЯ · ПОЛЯ".center(58) + "│")
    print("│" + "  + ЭНТРОПИЙНЫЙ РЕГУЛЯТОР".center(58) + "│")
    print("└" + "─" * 62 + "┘")


def _scale_pos(p, size):
    """Масштабирует каноническую позицию (из сетки 20³) на сетку size³."""
    k = size / 20.0
    return tuple(min(size - 1, max(0, int(round(c * k)))) for c in p)


def run_simulation(size: int = GRID_SIZE, seed: int = 42, quiet: bool = False,
                   animate: bool = False, bg_res: int = None) -> dict:
    """Полный сценарий (фазы 1–11).

    Фазы 1–8: вакуум → динамика частиц (бит-идентичны по seed;
    закреплены в tests/test_dynamics.py), мир `world`. Фазы 9–11
    (кварки и адроны, атом H, молекула H2) — в «лабораторных» мирах
    (свежий чистый вакуум на каждую фазу: в основном мире к t=810 чёрная
    дыра насытила объём, и связанная система там не уцелеет).
    Возвращает {'world', 'lab', 'frames'}; 'lab' — мир молекулы H2 (фаза 11).
    size — размер сетки (канонические позиции масштабируются из 20³);
    seed — сид RNG; quiet=True — без вывода в консоль.
    animate=True — фазы 9–11 пишут потоки лёгких кадров узлов (анимация):
    фаза 9 — каждый шаг, фазы 10/11 — каждый 2-й; по умолчанию False
    (11 кадров, как раньше).
    """
    if size < 4:
        raise ValueError(f"size: требуется ≥ 4, получено {size}")
    if quiet:
        import contextlib
        import io as _io

        with contextlib.redirect_stdout(_io.StringIO()):
            return _run_simulation(size, seed, animate, bg_res)
    return _run_simulation(size, seed, animate, bg_res)


def _run_simulation(size: int = GRID_SIZE, seed: int = 42,
                    animate: bool = False, bg_res: int = None) -> dict:
    """Полный сценарий: вакуум -> рождение -> взаимодействие -> сила -> регулятор -> чёрная дыра -> эволюция.

    bg_res — разрешение фона полных снапшотов (None → авто: полное для
    size<100, иначе наибольший делитель ≤64). Крупные сетки снимают
    снапшоты в numpy-массивах (snapshot_arrays) — list'ы из 10⁶ чисел
    съедают сотни МБ на кадр.
    """
    world = Volume3D(size=size, seed=seed)
    frames = []
    snap_bg = bg_res if bg_res is not None else _auto_bg(size)

    def _snap(v, label):
        if snap_bg is not None:
            return v.snapshot_arrays(label, bg_res=snap_bg)
        return v.snapshot(label)

    # --- ФАЗА 1: Вакуум (раслоение) ---
    print("\n" + "━" * 60)
    print("  ФАЗА 1: ВАКУУМ (раслоение, 20 шагов)")
    print("━" * 60)
    for _ in range(20):
        world.step()
    frames.append(world.snapshot("Фаза 1: вакуум (раслоение)"))
    print(world.visualize_layers())
    print(world.visualize_slice(axis=1, index=size // 2))

    # --- ФАЗА 2: Рождение частиц ---
    print("\n" + "━" * 60)
    print("  ФАЗА 2: РОЖДЕНИЕ ЧАСТИЦ (впрыск энергии)")
    print("━" * 60)
    positions = [_scale_pos(p, size) for p in ((5, 10, 8), (14, 10, 12), (10, 10, 10))]
    for pos in positions:
        world.inject_particle(pos)
        print(f"  -> Частица в {pos}")
    for _ in range(30):
        world.step()
    frames.append(_snap(world, "Фаза 2: рождение частиц"))
    print(world.visualize_layers())
    print(world.visualize_slice(axis=1, index=size // 2))

    # --- ФАЗА 3: Взаимодействие ---
    print("\n" + "━" * 60)
    print("  ФАЗА 3: ВЗАИМОДЕЙСТВИЕ (гравитация + поля, 40 шагов)")
    print("━" * 60)
    for _ in range(40):
        world.step()
    frames.append(_snap(world, "Фаза 3: взаимодействие"))
    print(world.visualize_layers())
    print(world.visualize_3d_preview(n_slices=3))

    # --- ФАЗА 4: Внешняя сила (деформация, «ломаем дерево») ---
    print("\n" + "━" * 60)
    print("  ФАЗА 4: ВНЕШНЯЯ СИЛА (деформация, аналог «ломаем дерево»)")
    print("━" * 60)
    force_pos = _scale_pos((10, 10, 10), size)
    world.apply_force(pos=force_pos, magnitude=0.6)
    print(f"  -> Приложена сила к центру {force_pos}")
    for _ in range(10):
        world.step()
    frames.append(_snap(world, "Фаза 4: внешняя сила (деформация)"))
    print(world.visualize_layers())
    print(world.visualize_slice(axis=1, index=size // 2))

    # --- ФАЗА 5: Энтропийный регулятор ---
    print("\n" + "━" * 60)
    print("  ФАЗА 5: ЭНТРОПИЙНЫЙ РЕГУЛЯТОР (20 шагов без + 50 с)")
    print("━" * 60)
    world.entropy_regulator_active = False
    for _ in range(20):
        world.step()
    entropy_without = world.global_entropy
    print(f"  Энтропия БЕЗ регулятора: {entropy_without:.4f}")

    world.entropy_regulator_active = True
    print(f"  Целевая энтропия: {ENTROPY_TARGET}")
    print(f"  Сила давления: {ENTROPY_PRESSURE}")
    for _ in range(50):
        world.step()
    entropy_with = world.global_entropy
    print(f"  Энтропия С регулятором:  {entropy_with:.4f}")
    print(f"  Изменение: {entropy_with - entropy_without:+.4f}")
    frames.append(_snap(world, "Фаза 5: энтропийный регулятор"))
    print(world.visualize_layers())

    # --- ФАЗА 6: Чёрная дыра (экстремальная гравитация) ---
    print("\n" + "━" * 60)
    print("  ФАЗА 6: ЧЁРНАЯ ДЫРА (концентрация энергии, 40 шагов)")
    print("━" * 60)
    bh_pos = _scale_pos((10, 10, 10), size)
    world._apply_injection({
        "pos": bh_pos, "energy": 8.0, "charge": 0.0, "radius": 2,
    })
    print(f"  -> Концентрация энергии в {bh_pos}")
    for _ in range(40):
        world.step()
    # Локальная диагностика: как гравитация замедляет время
    r = 2
    x0, y0, z0 = bh_pos
    box_mask = np.ones(world.n, dtype=bool)
    for axis in range(3):
        lo = max(0, [x0, y0, z0][axis] - r)
        hi = min(world.size, [x0, y0, z0][axis] + r + 1)
        m = np.ones(world.shape, dtype=bool)
        m[(slice(None),) * axis + (slice(lo, hi),)] = False
        box_mask &= m.reshape(world.n)
    if not box_mask.any():  # малая сетка: дополнение к ящику пусто → вся сетка
        box_mask = np.ones(world.n, dtype=bool)
    min_rate = float(world.time_rate[box_mask].min())
    min_grav = float(world.grav[box_mask].min())
    print(f"  Мин. скорость времени у «горизонта»: {min_rate:.3f} (в вакууме ~1.0)")
    print(f"  Мин. грав. потенциал (глубина ямы):  {min_grav:.3f} (насыщение -1.0)")
    frames.append(_snap(world, "Фаза 6: чёрная дыра"))
    print(world.visualize_layers())
    print(world.visualize_slice(axis=1, index=size // 2))

    # --- ФАЗА 7: Долгая эволюция ---
    print("\n" + "━" * 60)
    print("  ФАЗА 7: ДОЛГАЯ ЭВОЛЮЦИЯ (100 шагов, всё вместе)")
    print("━" * 60)
    for t in range(100):
        if t % 25 == 0 and t > 0:
            print(f"  t={world.time_step}: S={world.global_entropy:.4f}, "
                  f"E={world.total_energy:.2f}, Материя={world.matter_count}")
        world.step()
    frames.append(_snap(world, "Фаза 7: долгая эволюция"))

    # --- Фаза 8: ДИНАМИКА ЧАСТИЦ (спин, Pauli, распады, струны) ---
    print("\n" + "━" * 60)
    print("  ФАЗА 8: ДИНАМИКА ЧАСТИЦ (спин, Pauli, распады, струны, 500 шагов)")
    print("━" * 60)
    world.inject_particle(_scale_pos((4, 4, 4), size), "τ⁻", spin=0)
    world.inject_particle(_scale_pos((15, 15, 15), size), "μ⁻", spin=1)
    world.inject_particle(_scale_pos((8, 16, 8), size), "e⁻", spin=0)
    world.inject_quark_pair(_scale_pos((12, 6, 12), size), dist=2)
    world.inject_particle(_scale_pos((6, 6, 6), size), "n", spin=1)
    blocked = not world.inject_particle(_scale_pos((6, 6, 6), size), "n", spin=1)  # Pauli
    world.inject_particle(_scale_pos((7, 6, 6), size), "n", spin=0)
    world.inject_particle(_scale_pos((7, 6, 6), size), "n", spin=1)
    for _ in range(3):
        world.inject_particle(_scale_pos((9, 6, 6), size), "γ")  # бозе-стек (3 в ячейку)
    print(f"  τ⁻, μ⁻, e⁻ введены; струна u—ū (d=2); "
          f"Pauli: 2-й n↑ заблокирован={blocked}; 3 γ в стеке")
    for t in range(500):
        if t % 100 == 99:
            print(f"  t={world.time_step}: событий={len(world.decay_events)}, "
                  f"струн={len(world.strings)}, Pauli={world.pauli_blocks}, "
                  f"узлов={world.matter_count}")
        world.step()
    frames.append(_snap(world,
        "Фаза 8: динамика частиц (спин, Pauli, распады, струны)"))
    print(f"  Итог фазы: событий={len(world.decay_events)}, "
          f"разрывов струн={world.n_strings_broken}, "
          f"Pauli-блоков={world.pauli_blocks}")

    # --- ЛАБОРАТОРИЯ (фазы 9–11): чистый вакуум, по миру на фазу ---
    # В основном мире к t=810 чёрная дыра насытила объём (90–100% ячеек),
    # и связанная система там не уцелеет; осколки предыдущей фазы
    # (спонтанные рождения, движущиеся заряды) разрушают и следующую
    # систему. Поэтому каждая лабораторная фаза — свежий детерминированный
    # Volume3D (тот же size, seed; EM-физика без RNG). В результате
    # 'lab' = мир молекулы H2 (фаза 11).
    lab9 = Volume3D(size=size, seed=seed)

    # --- Фаза 9: КВАРКИ И АДРОНЫ (параметры PDG, конфайнмент, бюджет массы) ---
    print("\n" + "━" * 60)
    print("  ФАЗА 9: КВАРКИ И АДРОНЫ (параметры PDG, конфайнмент, 30 шагов) [лаб]")
    print("━" * 60)
    print(f"  Параметры (PDG 2024, масштаб S = 1.2/938.272 = "
          f"{NODE_MASS_SCALE:.2e} ед./эВ):")
    for q in ("u", "d", _ANTI_U, _ANTI_D):
        p = _particle(q)
        print(f"    {q}: m = {p.mass_MeV:.4g} МэВ, q = {_NODE_CHARGE[q]:+g}·e, "
              f"m_eff = {_NODE_MEFF[q]:.3e}·m_p")
    print(f"    p (uud): кварки {2 * 2.2 + 4.7:.1f} МэВ (1.0%) + "
          f"QCD-поле {938.27208816 - 2 * 2.2 - 4.7:.1f} МэВ (99.0%)")
    print(f"    E_rel: p = {HADRON_RELEASE['p']:.4f}, "
          f"n = {HADRON_RELEASE['n']:.4f} (модельных единиц)")
    quark_offs = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    qbase = lab9.find_free_center(_scale_pos((5, 12, 5), size),
                                  offsets=quark_offs)
    if qbase is not None:
        lab9.inject_particle(qbase, "u", spin=1)
        lab9.inject_particle((qbase[0] + 1, qbase[1], qbase[2]), "u", spin=0)
        lab9.inject_particle((qbase[0], qbase[1] + 1, qbase[2]), "d", spin=1)
    ok_p = lab9.hadronize()
    nbase = lab9.find_free_center(_scale_pos((3, 16, 3), size),
                                  offsets=quark_offs)
    if nbase is not None:
        lab9.inject_particle(nbase, "u", spin=0)
        lab9.inject_particle((nbase[0] + 1, nbase[1], nbase[2]), "d", spin=1)
        lab9.inject_particle((nbase[0], nbase[1] + 1, nbase[2]), "d", spin=0)
    ok_n = lab9.hadronize()
    lab9.inject_quark_pair(_scale_pos((10, 14, 6), size), dist=3, flavor="d")
    print(f"  uud → p: {ok_p};  udd → n: {ok_n};  d—d̄ струна (конфайнмент, d=3)")
    for _ in range(30):
        lab9.step()
        if animate:
            frames.append(lab9.snapshot_nodes(
                f"Фаза 9: кварки → адроны (лаб) · шаг {lab9.time_step}/30"))
    if not animate:
        frames.append(_snap(lab9, "Фаза 9: кварки и адроны (лаб)"))

    # --- Фаза 10: АТОМ (кулонова орбита электрона) ---
    print("\n" + "━" * 60)
    print("  ФАЗА 10: АТОМ H (кулон: e⁻ на круговой орбите, 250 шагов)")
    print("━" * 60)
    lab_atom = Volume3D(size=size, seed=seed)
    lab_atom.em_dynamics = True
    r0h = max(2, int(round(size / 7)))
    hcenter = lab_atom.find_free_center(_scale_pos((13, 10, 5), size),
                                        offsets=((0, 0, 0), (r0h, 0, 0)))
    if hcenter is not None:
        h = lab_atom.inject_hydrogen(hcenter, r0=r0h)
        print(f"  p + e⁻ (r0 = {r0h:.0f} ячеек), v_c = "
              f"{lab_atom._orbital_velocity(r0h):.3f} ячеек/шаг, "
              f"центр = {hcenter}")
        rmin_h, rmax_h = float("inf"), 0.0
        h_alive = True
        for _ in range(250):
            lab_atom.step()
            if animate and lab_atom.time_step % 2 == 0:
                frames.append(lab_atom.snapshot_nodes(
                    f"Фаза 10: атом H (лаб) · шаг {lab_atom.time_step}/250"))
            try:
                r = lab_atom.node_distance(h["p"], h["e"])
            except RuntimeError:
                h_alive = False
                break
            rmin_h, rmax_h = min(rmin_h, r), max(rmax_h, r)
        if h_alive:
            stable_h = (rmin_h >= 0.5 * h["r0"]) and (rmax_h <= 2.2 * h["r0"])
            print(f"  r(t): min {rmin_h:.2f} / max {rmax_h:.2f} "
                  f"(r0 = {h['r0']:.0f}) -> "
                  f"{'СТАБИЛЬНО (связанное состояние)' if stable_h else 'нестабильно'}")
        else:
            print("  узел атома распался (нестабильно)")
    else:
        print("  свободных ячеек для атома нет — фаза пропущена")
    if not animate:
        frames.append(_snap(lab_atom, "Фаза 10: атом H (кулон-орбита, лаб)"))

    # --- Фаза 11: МОЛЕКУЛА H2 ---
    print("\n" + "━" * 60)
    print("  ФАЗА 11: МОЛЕКУЛА H2 (2p + 2e⁻, 250 шагов) [лаб]")
    print("━" * 60)
    lab_h2 = Volume3D(size=size, seed=seed)
    lab_h2.em_dynamics = True
    d0m = max(2, int(round(size / 10)))
    r0m = max(2, int(round(size / 7)))
    mcenter = lab_h2.find_free_center(_scale_pos((17, 3, 14), size),
                                      offsets=((0, 0, 0), (d0m, 0, 0),
                                               (d0m // 2, 0, 0)))
    if mcenter is not None:
        m2 = lab_h2.inject_h2(mcenter, d0=d0m, r0=r0m)
        print(f"  p1—p2 (d0 = {d0m:.0f}), оба e⁻ в середине связи, "
              f"центр = {mcenter}")
        dmin_m, dmax_m = float("inf"), 0.0
        ermax_m = 0.0
        m_alive = True
        for _ in range(250):
            lab_h2.step()
            if animate and lab_h2.time_step % 2 == 0:
                frames.append(lab_h2.snapshot_nodes(
                    f"Фаза 11: молекула H2 (лаб) · шаг {lab_h2.time_step}/250"))
            try:
                p1n, p2n = lab_h2.find_node(m2["p1"]), lab_h2.find_node(m2["p2"])
                e1n, e2n = lab_h2.find_node(m2["e1"]), lab_h2.find_node(m2["e2"])
                if None in (p1n, p2n, e1n, e2n):
                    m_alive = False
                    break
                d = lab_h2.torus_distance(p1n["pos"], p2n["pos"])
                dmin_m, dmax_m = min(dmin_m, d), max(dmax_m, d)
                for e in (e1n, e2n):
                    er = min(lab_h2.torus_distance(e["pos"], p1n["pos"]),
                             lab_h2.torus_distance(e["pos"], p2n["pos"]))
                    ermax_m = max(ermax_m, er)
            except Exception:
                m_alive = False
                break
        if m_alive:
            # Связь осциллирует в 0.5–1.0·d0 (электроны сжимают ядра, но
            # не сливают в одну ячейку) — порог 0.45·d0 = «ядра не слились».
            stable_m = (dmin_m >= 0.45 * m2["d0"]) and (dmax_m <= 1.6 * m2["d0"]) \
                and (ermax_m <= 2.5 * m2["r0"])
            print(f"  p–p: min {dmin_m:.2f} / max {dmax_m:.2f} "
                  f"(d0 = {m2['d0']:.0f}); e–p max {ermax_m:.2f} -> "
                  f"{'СТАБИЛЬНО (связь удержана)' if stable_m else 'нестабильно'}")
        else:
            print("  узел молекулы распался (нестабильно)")
    else:
        print("  свободных ячеек для молекулы нет — фаза пропущена")
    if not animate:
        frames.append(_snap(lab_h2, "Фаза 11: молекула H2 (лаб)"))

    print(f"\n  ИТОГ: t={world.time_step}")
    print(world.visualize_layers())
    print(world.visualize_3d_preview(n_slices=3))
    print(world.visualize_slice(axis=1, index=size // 2))

    # --- График энтропии ---
    print("\n" + "━" * 60)
    print("  ГРАФИК ЭНТРОПИИ (последние 50 шагов)")
    print("━" * 60)
    _plot_entropy(world.entropy_history[-50:])

    # --- Консервация энергии ---
    print("\n  КОНСЕРВАЦИЯ ЭНЕРГИИ (Нётер):")
    print(f"    Суммарная энергия в начале:  {world.energy_history[0]:.2f}")
    print(f"    Суммарная энергия в конце:   {world.energy_history[-1]:.2f}")
    print(f"    Дрейф (впрыски + регулятор): {world.energy_history[-1] - world.energy_history[0]:+.2f}")

    print(f"\n  ЛАБОРАТОРИЯ (чистый вакуум): H2-мир t={lab_h2.time_step}, "
          f"узлов={lab_h2.matter_count} (кварки → адроны → H → H2)")

    return {"world": world, "lab": lab_h2, "frames": frames}


def _plot_entropy(history: list, width: int = 50):
    """ASCII график энтропии."""
    if not history:
        return
    n = len(history)
    step = max(1, n // width)
    sampled = history[::step]
    if not sampled:
        return

    max_val = max(sampled) * 1.2
    min_val = min(sampled) * 0.8
    range_val = max_val - min_val if max_val != min_val else 1.0

    height = 15
    print(f"\n  {max_val:.4f} ┤")
    target_row = ((ENTROPY_TARGET - min_val) / range_val * height) if min_val <= ENTROPY_TARGET <= max_val else -1
    for row in range(height - 1, -1, -1):
        threshold = min_val + (range_val * row / height)
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            elif val >= threshold - range_val / height:
                line += "▄"
            else:
                line += " "
        marker = " ← ЦЕЛЬ" if abs(row - target_row) < 0.5 else ""
        print(f"  {threshold:.4f} ┤{line}{marker}")
    print(f"  {min_val:.4f} ┤{'─' * len(sampled)}")
    print(f"             └{'─' * (len(sampled) + 1)}")
    print(f"               0{' ' * (len(sampled) - 3)}{len(history)}")


# ============================================================
# ЭКСПОРТ ДЛЯ 3D-ВЬЮВЕРА
# ============================================================

# Бинарный формат полных кадров (frames.bin), D³ на поле, фиксированный порядок:
BIN_F32 = ("time_phase", "time_rate", "energy", "psi_abs", "psi_phase",
           "grav", "field_mag", "fx", "fy", "fz")   # 10 × float32
BIN_U8 = ("matter", "spin", "stack")                # 3 × uint8 (spin: 0=нет, 1=↓, 2=↑)
_SINGLE_FILE_MAX = 32 * 1024 * 1024  # бинарник ≤32 МБ → встраивается в HTML (base64)


def downsample(arr, size: int, D: int, mode: str = "mean"):
    """Даунсемпл поля S³ до D³ (D должен делить size): mean для полей,
    max — для дискретных (matter/stack/spin). arr: (S,S,S) или (3,S,S,S)."""
    if size % D:
        raise ValueError(f"bg_res={D} не делит размер сетки {size}")
    k = size // D
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape == (size, size, size):
        a, axes = arr.reshape(D, k, D, k, D, k), (1, 3, 5)
    elif arr.ndim == 4 and arr.shape == (3, size, size, size):
        a, axes = arr.reshape(3, D, k, D, k, D, k), (2, 4, 6)
    else:
        raise ValueError(f"downsample: ожидалось (S,S,S) или (3,S,S,S), "
                         f"получено {arr.shape}")
    if mode == "mean":
        return a.mean(axis=axes)
    if mode == "max":
        return a.max(axis=axes)
    raise ValueError(f"downsample: неизвестный режим {mode!r}")


def _auto_bg(size: int):
    """Авто-разрешение фона: size ≤ 64 → полное (None); иначе наибольший
    делитель size ≤ 64 (≤ 64³ = 262 тыс. ячеек). 80→40, 100→50, 1000→50."""
    if size <= 64:
        return None
    divs = [d for d in range(1, size + 1) if size % d == 0 and d <= 64]
    return divs[-1] if divs else None


def _frame_arrays(frame: dict) -> dict:
    """Полный кадр → dict numpy-массивов в разрешении кадра (работает и с
    snapshot() — list'ы, и с snapshot_arrays() — массивы, уже даунсемплены)."""
    d = frame["data"]
    n = np.asarray(d["energy"]).size
    S = int(round(n ** (1.0 / 3.0)))
    if S ** 3 != n:
        raise ValueError(f"данные кадра не куб: {n} значений")
    out = {}
    for name in ("time_phase", "time_rate", "energy", "psi_abs",
                 "psi_phase", "grav", "field_mag"):
        out[name] = np.asarray(d[name], dtype=np.float64).reshape(S, S, S)
    out["field_vec"] = np.asarray(d["field_vec"], dtype=np.float64) \
        .reshape(3, S, S, S)
    for name in ("matter", "spin", "stack"):
        out[name] = np.asarray(d[name], dtype=np.int8).reshape(S, S, S)
    return out


def export_viewer(frames: list, out_dir: str, size: int = GRID_SIZE,
                  bg_res: int = None, single_file: bool = None):
    """Записывает snapshots.json + viewer.html (+ frames.bin в файловом режиме).

    Полные кадры → бинарник (10×float32 + 3×uint8 на D³, см. BIN_F32/BIN_U8)
    + min/max по полям; JSON несёт метаданные, точные узлы/струны/распады.
    bg_res — разрешение фона D (по умолчанию авто, см. _auto_bg).
    single_file=None → если бинарник ≤ 32 МБ, base64 в одном viewer.html
    (открывается двойным кликом); иначе frames.bin рядом (нужен `wm serve`).
    Шаблон ищется в out_dir, затем в корне проекта.
    """
    os.makedirs(out_dir, exist_ok=True)
    D = bg_res if bg_res is not None else _auto_bg(size)

    payload_frames = []
    bin_chunks = []
    offset = 0
    _fv = {"fx": 0, "fy": 1, "fz": 2}
    for f in frames:
        if f.get("light"):
            payload_frames.append({
                "t": f["t"], "label": f["label"], "light": True,
                "metrics": f["metrics"], "layers": f["layers"],
                "data": {k: f["data"].get(k, [])
                         for k in ("nodes", "strings", "decays")},
            })
            continue
        arr = _frame_arrays(f)
        Sd = arr["energy"].shape[0]
        if D is None:
            D = Sd
        elif Sd != D:
            for name in list(arr):
                mode = "max" if name in ("matter", "spin", "stack") else "mean"
                arr[name] = downsample(arr[name], Sd, D, mode=mode)
        # --- бинарный чанк кадра ---
        chunk = bytearray()
        minmax = {}
        for name in BIN_F32:
            a = arr["field_vec"][_fv[name]] if name in _fv else arr[name]
            a32 = np.ascontiguousarray(a, dtype=np.float32).ravel()
            minmax[name] = [float(a32.min()), float(a32.max())]
            chunk += a32.tobytes()
        spin_enc = np.where(arr["spin"] < 0, 0, arr["spin"] + 1).astype(np.uint8)
        _u8 = {"matter": arr["matter"], "spin": spin_enc, "stack": arr["stack"]}
        for name in BIN_U8:
            chunk += np.ascontiguousarray(_u8[name], dtype=np.uint8) \
                .ravel().tobytes()
        bin_chunks.append(bytes(chunk))
        payload_frames.append({
            "t": f["t"], "label": f["label"],
            "metrics": f["metrics"], "layers": f["layers"],
            "bin": [offset, len(chunk)],
            "minmax": minmax,
            "data": {k: f["data"].get(k, [])
                     for k in ("nodes", "strings", "decays")},
        })
        offset += len(chunk)

    payload = {
        "size": size,
        "bg": D if D is not None and D != size else None,
        "layers": LAYER_NAMES,
        "typeColors": [NODE_COLORS.get(c, [1.0, 1.0, 1.0])
                       for c in range(len(TYPE_CODE))],
        "frames": payload_frames,
    }
    bin_bytes = b"".join(bin_chunks)
    embed = single_file if single_file is not None \
        else (len(bin_bytes) <= _SINGLE_FILE_MAX)
    if embed:
        import base64
        payload["bin_b64"] = base64.b64encode(bin_bytes).decode("ascii")
        print(f"\n  Фон: один файл (base64, {len(bin_bytes) / 1024:.0f} КБ)")
    else:
        payload["bin"] = "frames.bin"
        with open(os.path.join(out_dir, "frames.bin"), "wb") as bf:
            bf.write(bin_bytes)
        print(f"\n  Фон: frames.bin ({len(bin_bytes) / 1048576:.1f} МБ) — "
              f"viewer.html требует `wm serve`")
    if D is not None and D != size:
        print(f"  Разрешение фона: {D}³ (полное {size}³), узлы — точные")
    data_json = json.dumps(payload, separators=(",", ":"))

    json_path = os.path.join(out_dir, "snapshots.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(data_json)
    print(f"\n  Данные: {json_path} ({os.path.getsize(json_path) / 1024:.0f} КБ)")

    tpl_path = os.path.join(out_dir, "viewer_template.html")
    if not os.path.isfile(tpl_path):
        # fallback: корень проекта (родитель каталога пакета wm/)
        alt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "viewer_template.html",
        )
        if os.path.isfile(alt):
            tpl_path = alt
    if not os.path.isfile(tpl_path):
        raise FileNotFoundError(
            "viewer_template.html не найден (в out_dir и корне проекта)")
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = f.read()
    marker = "/*__DATA__*/null"
    if marker not in tpl:
        raise RuntimeError("В шаблоне нет маркера /*__DATA__*/null")
    html = tpl.replace(marker, data_json)
    html_path = os.path.join(out_dir, "viewer.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  3D-вьювер: {html_path} ({os.path.getsize(html_path) / 1024:.0f} КБ) — открыть в браузере")


# ============================================================
# ЗАПУСК
# ============================================================

def main(out_dir=None, size: int = GRID_SIZE, seed: int = 42,
         animate: bool = False, bg_res: int = None) -> dict:
    """Точка входа: баннер → сценарий → итог → snapshots.json + viewer.html.

    out_dir=None — текущий каталог; animate=True — анимация фаз 9–11
    (потоки лёгких кадров узлов); bg_res — разрешение фона (D³), по
    умолчанию авто (см. _auto_bg).
    """
    import sys
    import time as _time

    # UTF-8 вывод (консоль Windows по умолчанию cp1251)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    _banner()
    t0 = _time.time()
    result = run_simulation(size=size, seed=seed, animate=animate,
                            bg_res=bg_res)
    dt = _time.time() - t0
    world = result["world"]

    print("\n" + "═" * 60)
    print("  МОДЕЛЬ ЗАВЕРШЕНА")
    print("═" * 60)
    print(f"""
  ВРЕМЯ ВЫЧИСЛЕНИЯ: {dt:.1f} с ({world.time_step} шагов, {world.n} ячеек)

  СВОЙСТВА СУБПРОСТРАНСТВ (из экспериментов):

  L0 ВРЕМЯ:
    • Направленность: всегда → (стрелка времени)
    • Дилатация: замедление в грав. яме (ОТО, GPS +38.5 мкс/день: +45.7 грав, −7.2 кинем)
    • Причинный радиус: c (Hafele-Keating, мюоны)

  L1 ЭНЕРГИЯ:
    • Консервация: ΣE = const (Нётер)
    • Квантование: ΔE ≥ hbar (спектры атомов)
    • E=mc²: аннигиляция e⁺e⁻ → 2γ (100%)
    • Нулевая: E₀ ≠ 0 (Казимир, LIGO)

  L2 ВЕРОЯТНОСТЬ:
    • Суперпозиция: двойной щель (Tonomura 1989)
    • Нелокальность: Белл нарушен (Aspect 1982)
    • Неопределённость: Δx·Δp ≥ hbar/2
    • Коллапс: при измерении (Копенгаген)

  L3 ГРАВИТАЦИЯ:
    • Кривизна ∝ T_μν (Эйнштейн, LIGO 2015)
    • Притяжение: всегда (нет отрицательной массы)
    • Слабость: 10⁻³⁶ от ЭМ
    • Скорость: c (грав. волны, GW150914)

  L4 ПОЛЯ:
    • Диполь: +/− (Штерн-Герлах 1922)
    • Сила: сильная на макро (магнит держит гвоздь)
    • Бесконечный радиус: фотон безмассовый
    • Калибровка: U(1), спин=1 (QED)

  ЭНТРОПИЙНЫЙ РЕГУЛЯТОР:
    • Мониторит S(t), применяет давление P
    • S > target → охлаждение (синтез)
    • S < target → нагрев (раслоение)
    • Аналог: термостат / тёмная энергия

  ДИНАМИКА ЧАСТИЦ (решётка, фаза 8):
    • Спин ↑/↓: 2 слота на ячейку (принцип Pauli)
    • Ферми/Бозе: фермион — 1 на спин-слот; бозон — стек ≤ 3 узла/ячейку
    • Цепочки распадов: τ⁻ → μ⁻/π± → … → e⁻ (стабильна); τ лог-сжаты
    • Струны: u—ū конфайнмент (σ·d), разрыв при d≥4, аннигиляция при d≤1

  КВАРКИ, АТОМЫ И МОЛЕКУЛЫ (фазы 9–11):
    • Параметры кварков: массы/заряды PDG 2024; масштаб S = 1.2/938.272 ед./эВ
    • Хадронизация: uud→p, udd→n (R ≤ 2 ячейки); избыток массы (99% = QCD-поле)
      депонируется как энергия поля (ledger 'hadronization')
    • ЭМ-динамика (em_dynamics): F = K·q₁q₂·r̂/(d²+ε²), симплициев Евлер, тор
    • Атом H: e⁻ на круговой кулоновой орбите (v_c = √(K/m_e·r)), стабильность
      проверяется по r(t) за 250 шагов
    • Молекула H2: связь p–p + 2 e⁻, стабильность: d(p–p) и e–p за 250 шагов
""")

    export_viewer(result["frames"], out_dir, size=size, bg_res=bg_res)
    return result


if __name__ == "__main__":
    main()
