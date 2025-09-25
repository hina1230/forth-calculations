#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FORTH Theory Article #002: Magnetic Field Visualization
Accurate magnetic field calculation in 4D torus structure
Author: Yoshiyuki Matsuyama
Date: 2025-09-25
"""

import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime

class MagneticFieldSimulation:
    """4次元トーラス構造における磁場の正確な計算"""

    def __init__(self):
        # 物理定数（CODATA 2018値）
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.c = 299792458    # m/s (exact)
        self.M_sun = 1.98847e30  # kg

        # M87ブラックホールパラメータ
        self.M_M87_solar = 6.5e9  # 太陽質量単位
        self.M_M87 = self.M_M87_solar * self.M_sun  # kg

        # シュワルツシルト半径を計算
        self.Rs = 2 * self.G * self.M_M87 / (self.c ** 2)

        # トーラスパラメータ（Rs単位）
        self.R_over_r = 1000  # 主半径と管半径の比
        self.R = 1000 * self.Rs  # 主半径 (m)
        self.r = self.Rs  # 管半径 (m)

        # グリッド設定
        self.n_theta = 50  # ポロイダル角解像度
        self.n_phi = 50    # トロイダル角解像度
        self.n_w = 8       # W軸位相解像度

        # 結果を格納
        self.results = {}

    def calculate_jet_velocity(self):
        """正確なジェット速度の計算"""
        r_over_R = 1 / self.R_over_r
        v_over_c = np.sqrt(1 - r_over_R**2)

        calculation_log = {
            "formula": "v/c = sqrt(1 - (r/R)^2)",
            "inputs": {
                "R/r": self.R_over_r,
                "r/R": r_over_R
            },
            "steps": [
                f"r/R = 1/{self.R_over_r} = {r_over_R:.6f}",
                f"(r/R)^2 = {r_over_R**2:.12f}",
                f"1 - (r/R)^2 = {1 - r_over_R**2:.12f}",
                f"v/c = sqrt({1 - r_over_R**2:.12f}) = {v_over_c:.9f}"
            ],
            "result": {
                "v_over_c": v_over_c,
                "gamma_factor": 1 / np.sqrt(1 - v_over_c**2)
            }
        }

        self.results['jet_velocity'] = calculation_log
        return v_over_c

    def calculate_magnetic_field_structure(self):
        """磁場構造の計算（透明性重視）"""

        # 座標グリッドの生成
        theta = np.linspace(0, 2*np.pi, self.n_theta)
        phi = np.linspace(0, 2*np.pi, self.n_phi)
        w_phase = np.linspace(0, 2*np.pi, self.n_w)

        # 3次元メッシュグリッド
        THETA, PHI = np.meshgrid(theta, phi)

        # 磁場強度の基準値（観測的推定値から）
        B0 = 100  # Gauss (M87での典型値)

        magnetic_field_data = []

        for w in w_phase:
            # トロイダル成分（主磁場）
            B_toroidal = B0 * np.sqrt(self.r / self.R)

            # ポロイダル成分（W軸効果）
            B_poloidal = 0.1 * B0 * np.sin(w)

            # W軸成分（4次元効果）
            B_w = 0.05 * B0 * np.cos(PHI + w)

            # 全磁場強度
            B_total = np.sqrt(B_toroidal**2 + B_poloidal**2 + B_w**2)

            magnetic_field_data.append({
                "w_phase": float(w),
                "B_toroidal": float(B_toroidal),
                "B_poloidal": float(B_poloidal),
                "B_w_max": float(np.max(np.abs(B_w))),
                "B_total_mean": float(np.mean(B_total)),
                "B_total_max": float(np.max(B_total)),
                "B_total_min": float(np.min(B_total))
            })

        # 螺旋構造パラメータ
        pitch_angle = np.arctan(self.r / self.R)
        helical_turns = 2 * np.pi * self.R / self.r

        field_analysis = {
            "reference_field_strength": B0,
            "pitch_angle_rad": pitch_angle,
            "pitch_angle_deg": np.degrees(pitch_angle),
            "helical_turns": helical_turns,
            "characteristic_scale": 2 * np.pi * self.r / self.Rs,
            "magnetic_field_data": magnetic_field_data
        }

        self.results['magnetic_field_structure'] = field_analysis
        return field_analysis

    def calculate_energy_distribution(self):
        """エネルギー分布の計算"""

        # 磁場エネルギー密度 (SI単位)
        B0 = 0.01  # Tesla (100 Gauss)
        mu0 = 4 * np.pi * 1e-7  # 真空透磁率
        U_B = B0**2 / (2 * mu0)  # J/m^3

        # プラズマパラメータ（観測的推定）
        n_e = 1e8  # 電子数密度 /m^3
        m_e = 9.109e-31  # 電子質量 kg
        v_jet = self.calculate_jet_velocity() * self.c

        # 運動エネルギー密度
        U_K = 0.5 * n_e * m_e * v_jet**2  # J/m^3

        # エネルギー変換効率
        efficiency = U_K / (U_B + U_K)

        energy_log = {
            "magnetic_energy_density": U_B,
            "kinetic_energy_density": U_K,
            "total_energy_density": U_B + U_K,
            "conversion_efficiency": efficiency,
            "plasma_beta": 2 * mu0 * n_e * m_e * self.c**2 / B0**2,
            "detailed_calculation": {
                "B_field_Tesla": B0,
                "electron_density_per_m3": n_e,
                "jet_velocity_m_per_s": v_jet,
                "vacuum_permeability": mu0
            }
        }

        self.results['energy_distribution'] = energy_log
        return energy_log

    def generate_3d_visualization(self):
        """3D可視化データの生成"""

        # トーラス表面の座標計算
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, 2*np.pi, 100)
        U, V = np.meshgrid(u, v)

        # トーラスの3D座標（Rs単位で正規化）
        X = (self.R/self.Rs + (self.r/self.Rs) * np.cos(V)) * np.cos(U)
        Y = (self.R/self.Rs + (self.r/self.Rs) * np.cos(V)) * np.sin(U)
        Z = (self.r/self.Rs) * np.sin(V)

        # 磁場強度を色として表現
        B_normalized = np.sqrt((self.r/self.R) / (1 + (self.r/self.R) * np.cos(V)))

        # Plotlyフィギュアの作成
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=B_normalized,
                colorscale='Viridis',
                name='Magnetic Field Intensity',
                colorbar=dict(
                    title='B/B0'
                ),
                showscale=True
            )
        ])

        # 磁力線の追加（サンプル）
        n_lines = 20
        for i in range(n_lines):
            phi = 2 * np.pi * i / n_lines
            t = np.linspace(0, 4*np.pi, 200)

            # 螺旋磁力線
            x_line = (self.R/self.Rs + (self.r/self.Rs) * np.cos(t)) * np.cos(phi + 0.1*t)
            y_line = (self.R/self.Rs + (self.r/self.Rs) * np.cos(t)) * np.sin(phi + 0.1*t)
            z_line = (self.r/self.Rs) * np.sin(t)

            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='red', width=1),
                name=f'Field Line {i+1}',
                showlegend=False
            ))

        fig.update_layout(
            title={
                'text': 'FORTH Theory: 4D Torus Magnetic Field Structure',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title='X (Rs)',
                yaxis_title='Y (Rs)',
                zaxis_title='Z (Rs)',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )

        # HTMLファイルとして保存（オフライン版として作成）
        html_file = 'magnetic_field_3d.html'
        # 'directory'オプションを使用してPlotly.jsを別ファイルとして保存
        fig.write_html(html_file, include_plotlyjs='directory')

        self.results['visualization'] = {
            "file": html_file,
            "grid_points": 100 * 100,
            "field_lines": n_lines,
            "coordinate_system": "Schwarzschild radius units"
        }

        return fig

    def calculate_observational_predictions(self):
        """観測可能な予測の計算"""

        # W軸周期（消失時間）
        T_W = 2 * np.pi * self.r / self.c  # seconds
        T_W_hours = T_W / 3600  # hours

        # 特徴的周波数
        f_R = self.c / (2 * np.pi * self.R)  # 主半径周波数
        f_r = self.c / (2 * np.pi * self.r)  # 管半径周波数

        predictions = {
            "W_axis_period": {
                "seconds": T_W,
                "hours": T_W_hours,
                "days": T_W_hours / 24
            },
            "characteristic_frequencies": {
                "major_radius_Hz": f_R,
                "minor_radius_Hz": f_r,
                "ratio": f_r / f_R
            },
            "expected_polarization": {
                "pattern": "Helical with pitch angle",
                "pitch_angle_rad": np.arctan(self.r / self.R),
                "rotation_measure": "Variable with W-phase"
            }
        }

        self.results['observational_predictions'] = predictions
        return predictions

    def save_results(self):
        """計算結果をJSONファイルに保存"""
        output = {
            "metadata": {
                "calculation_date": datetime.now().isoformat(),
                "author": "Yoshiyuki Matsuyama",
                "article": "#002 Magnetic Field Visualization",
                "version": "1.0 (accurate)"
            },
            "physical_constants": {
                "G": self.G,
                "c": self.c,
                "M_sun": self.M_sun
            },
            "torus_parameters": {
                "R_over_r": self.R_over_r,
                "R_meters": self.R,
                "r_meters": self.r,
                "R_Rs_units": self.R / self.Rs,
                "r_Rs_units": self.r / self.Rs
            },
            "schwarzschild_radius": {
                "value_meters": self.Rs,
                "value_km": self.Rs / 1000,
                "value_AU": self.Rs / 1.496e11
            },
            "results": self.results
        }

        with open('calculation_results_002.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output

    def generate_report(self):
        """計算レポートの生成"""

        # 各計算の実行
        v_jet = self.calculate_jet_velocity()
        field = self.calculate_magnetic_field_structure()
        energy = self.calculate_energy_distribution()
        predictions = self.calculate_observational_predictions()

        report = f"""
# 計算検証レポート #002: 磁場構造

## 1. 基本パラメータ
シュワルツシルト半径: Rs = {self.Rs:.3e} m
主半径: R = {self.R_over_r} Rs = {self.R:.3e} m
管半径: r = 1 Rs = {self.r:.3e} m

## 2. ジェット速度（R/r = {self.R_over_r}）
v/c = {v_jet:.9f}
gamma = {self.results['jet_velocity']['result']['gamma_factor']:.1f}

## 3. 磁場構造
螺旋ピッチ角: {field['pitch_angle_deg']:.3f} degrees
螺旋巻き数: {field['helical_turns']:.0f}
特徴的スケール: {field['characteristic_scale']:.1f} Rs

## 4. エネルギー分析
磁場エネルギー密度: {energy['magnetic_energy_density']:.3e} J/m^3
運動エネルギー密度: {energy['kinetic_energy_density']:.3e} J/m^3
変換効率: {energy['conversion_efficiency']*100:.1f}%

## 5. 観測予測
W軸周期: {predictions['W_axis_period']['hours']:.1f} 時間
主半径周波数: {predictions['characteristic_frequencies']['major_radius_Hz']:.3e} Hz
管半径周波数: {predictions['characteristic_frequencies']['minor_radius_Hz']:.3e} Hz

## 検証完了
全計算は手計算で確認可能
        """

        return report


if __name__ == "__main__":
    # シミュレーション実行
    sim = MagneticFieldSimulation()

    # レポート生成
    report = sim.generate_report()
    print(report)

    # 3D可視化生成
    fig = sim.generate_3d_visualization()
    print("\n3D可視化をmagnetic_field_3d.htmlに保存しました")

    # 結果保存
    results = sim.save_results()
    print("計算結果をcalculation_results_002.jsonに保存しました")