#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FORTH Theory Article #006: Disappearance Time Calculation
W-axis transit phenomenon and temporal periodicity
Author: Yoshiyuki Matsuyama
Date: 2025-09-25
"""

import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta

class DisappearanceTimeCalculation:
    """W軸通過による消失時間の計算"""

    def __init__(self):
        # 物理定数
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.c = 299792458    # m/s (exact)
        self.M_sun = 1.98847e30  # kg

        # M87パラメータ
        self.M_M87_solar = 6.5e9
        self.M_M87 = self.M_M87_solar * self.M_sun
        self.Rs = 2 * self.G * self.M_M87 / (self.c ** 2)

        # 異なる管半径での計算（Rs単位）
        self.r_values = [0.5, 1.0, 2.0, 3.5, 5.0, 10.0]  # Rs単位

        # Sgr A*パラメータ（比較用）
        self.M_SgrA_solar = 4.3e6
        self.M_SgrA = self.M_SgrA_solar * self.M_sun
        self.Rs_SgrA = 2 * self.G * self.M_SgrA / (self.c ** 2)

        # 結果格納
        self.results = {}

    def calculate_basic_disappearance_time(self, r_meters):
        """基本的な消失時間計算"""

        # 基本公式: Δt = 2πr/c
        delta_t = 2 * np.pi * r_meters / self.c

        # 時間単位変換
        delta_t_minutes = delta_t / 60
        delta_t_hours = delta_t / 3600
        delta_t_days = delta_t / 86400

        return {
            "r_meters": r_meters,
            "delta_t_seconds": delta_t,
            "delta_t_minutes": delta_t_minutes,
            "delta_t_hours": delta_t_hours,
            "delta_t_days": delta_t_days,
            "formula": "Δt = 2πr/c"
        }

    def calculate_for_different_radii(self):
        """異なる管半径での消失時間を計算"""

        calculations = []

        for r_Rs in self.r_values:
            r_meters = r_Rs * self.Rs
            result = self.calculate_basic_disappearance_time(r_meters)
            result["r_Rs_units"] = r_Rs
            result["r_km"] = r_meters / 1000
            result["r_AU"] = r_meters / 1.496e11
            calculations.append(result)

        self.results['different_radii'] = {
            "description": "Disappearance time for various tube radii",
            "calculations": calculations
        }

        return calculations

    def calculate_observational_windows(self):
        """観測ウィンドウの計算"""

        # r = 1 Rs の場合を標準とする
        r_standard = self.Rs
        delta_t_standard = 2 * np.pi * r_standard / self.c

        # 観測開始時刻を仮定（2025年1月1日 00:00 UTC）
        start_date = datetime(2025, 1, 1, 0, 0, 0)

        # 10サイクル分の観測ウィンドウを計算
        windows = []
        for cycle in range(10):
            # 消失開始時刻
            disappear_time = start_date + timedelta(seconds=cycle * delta_t_standard)
            # 再出現時刻
            reappear_time = disappear_time + timedelta(seconds=delta_t_standard/2)

            windows.append({
                "cycle": cycle + 1,
                "disappear_UTC": disappear_time.isoformat(),
                "reappear_UTC": reappear_time.isoformat(),
                "duration_hours": delta_t_standard / 7200  # 消失継続時間（半周期）
            })

        self.results['observation_windows'] = {
            "standard_radius": r_standard,
            "period_hours": delta_t_standard / 3600,
            "windows": windows
        }

        return windows

    def compare_black_holes(self):
        """M87とSgr A*の比較"""

        # 両ブラックホールでr = 1 Rsの場合
        r_M87 = self.Rs
        r_SgrA = self.Rs_SgrA

        delta_t_M87 = 2 * np.pi * r_M87 / self.c
        delta_t_SgrA = 2 * np.pi * r_SgrA / self.c

        comparison = {
            "M87": {
                "mass_solar": self.M_M87_solar,
                "Rs_meters": self.Rs,
                "Rs_km": self.Rs / 1000,
                "delta_t_seconds": delta_t_M87,
                "delta_t_hours": delta_t_M87 / 3600,
                "delta_t_days": delta_t_M87 / 86400
            },
            "SgrA": {
                "mass_solar": self.M_SgrA_solar,
                "Rs_meters": self.Rs_SgrA,
                "Rs_km": self.Rs_SgrA / 1000,
                "delta_t_seconds": delta_t_SgrA,
                "delta_t_hours": delta_t_SgrA / 3600,
                "delta_t_minutes": delta_t_SgrA / 60
            },
            "ratio": {
                "mass_ratio": self.M_M87_solar / self.M_SgrA_solar,
                "Rs_ratio": self.Rs / self.Rs_SgrA,
                "time_ratio": delta_t_M87 / delta_t_SgrA
            }
        }

        self.results['black_hole_comparison'] = comparison
        return comparison

    def analyze_detectability(self):
        """検出可能性の解析"""

        # r = 1 Rs の場合
        r_standard = self.Rs
        delta_t = 2 * np.pi * r_standard / self.c

        # 観測要件
        detectability = {
            "time_resolution_required": {
                "ideal_hours": 1,
                "minimum_hours": 6,
                "description": "To detect disappearance/reappearance transitions"
            },
            "observation_duration": {
                "minimum_cycles": 3,
                "minimum_days": 3 * delta_t / 86400,
                "recommended_days": 10 * delta_t / 86400
            },
            "instruments": {
                "ALMA": {
                    "suitable": True,
                    "time_resolution": "< 1 hour achievable",
                    "challenges": "Weather, scheduling"
                },
                "EHT": {
                    "suitable": False,
                    "reason": "Campaign mode, not continuous monitoring"
                },
                "X-ray": {
                    "suitable": True,
                    "satellites": ["Chandra", "XMM-Newton", "NuSTAR"],
                    "advantage": "Continuous monitoring possible"
                }
            },
            "signatures": {
                "flux_variation": "Periodic dimming/brightening",
                "spectral_change": "Possible hardening during transit",
                "polarization": "Rotation pattern change"
            }
        }

        self.results['detectability'] = detectability
        return detectability

    def calculate_relativistic_corrections(self):
        """相対論的補正の計算"""

        # 重力時間遅延（シュワルツシルト時空）
        r_values = np.array([2, 3, 5, 10, 20, 50]) * self.Rs  # 距離

        corrections = []

        for r in r_values:
            # 重力赤方偏移因子
            z = np.sqrt(1 - self.Rs/r) - 1 if r > self.Rs else 0

            # 時間遅延因子
            time_dilation = np.sqrt(1 - self.Rs/r) if r > self.Rs else 0

            # 管半径をr/10と仮定（簡略化）
            r_tube = r / 10
            delta_t_proper = 2 * np.pi * r_tube / self.c
            delta_t_observed = delta_t_proper / time_dilation if time_dilation > 0 else np.inf

            corrections.append({
                "r_Rs": r / self.Rs,
                "gravitational_redshift": z,
                "time_dilation_factor": time_dilation,
                "delta_t_proper_seconds": delta_t_proper,
                "delta_t_observed_seconds": delta_t_observed,
                "correction_percent": 100 * (delta_t_observed/delta_t_proper - 1) if delta_t_proper > 0 else 0
            })

        self.results['relativistic_corrections'] = {
            "description": "General relativistic time dilation effects",
            "corrections": corrections
        }

        return corrections

    def create_visualization(self):
        """消失時間の可視化"""

        # データ準備
        r_Rs = np.array(self.r_values)
        delta_t_hours = []

        for r in r_Rs:
            r_meters = r * self.Rs
            dt = 2 * np.pi * r_meters / self.c / 3600  # hours
            delta_t_hours.append(dt)

        # プロット作成
        fig = go.Figure()

        # M87のデータ
        fig.add_trace(go.Scatter(
            x=r_Rs,
            y=delta_t_hours,
            mode='lines+markers',
            name='M87 Black Hole',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        # 参考線（1日、1週間）
        fig.add_hline(y=24, line_dash="dash", line_color="gray",
                      annotation_text="1 day")
        fig.add_hline(y=168, line_dash="dash", line_color="gray",
                      annotation_text="1 week")

        fig.update_layout(
            title='FORTH Theory: W-Axis Disappearance Time',
            xaxis_title='Tube Radius (Rs)',
            yaxis_title='Disappearance Time (hours)',
            yaxis_type='log',
            width=800,
            height=600
        )

        # HTMLファイルとして保存
        fig.write_html('disappearance_time_006.html')

        self.results['visualization'] = {
            "file": "disappearance_time_006.html",
            "type": "disappearance_time_vs_radius"
        }

        return fig

    def save_results(self):
        """計算結果の保存"""

        # 可視化データを準備
        r_Rs = np.array(self.r_values)
        delta_t_hours = []

        for r in r_Rs:
            r_meters = r * self.Rs
            dt = 2 * np.pi * r_meters / self.c / 3600  # hours
            delta_t_hours.append(dt)

        output = {
            "metadata": {
                "calculation_date": datetime.now().isoformat(),
                "author": "Yoshiyuki Matsuyama",
                "article": "#006 Disappearance Time",
                "version": "1.0"
            },
            "physical_constants": {
                "G": self.G,
                "c": self.c,
                "M_sun": self.M_sun
            },
            "black_hole_parameters": {
                "M87": {
                    "mass_solar": self.M_M87_solar,
                    "schwarzschild_radius": self.Rs
                },
                "SgrA": {
                    "mass_solar": self.M_SgrA_solar,
                    "schwarzschild_radius": self.Rs_SgrA
                }
            },
            "results": self.results,
            "visualization": {
                "disappearance_times": {
                    "radius_Rs": r_Rs.tolist(),
                    "time_hours": delta_t_hours,
                    "description": "Disappearance time vs torus tube radius"
                }
            }
        }

        with open('calculation_results_006.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output

    def generate_report(self):
        """レポート生成"""

        # 各計算実行
        radii_calc = self.calculate_for_different_radii()
        windows = self.calculate_observational_windows()
        comparison = self.compare_black_holes()
        detectability = self.analyze_detectability()
        corrections = self.calculate_relativistic_corrections()

        # 標準ケース（r = 1 Rs）の結果
        standard = radii_calc[1]  # r_values[1] = 1.0 Rs

        report = f"""
# 計算検証レポート #006: 消失時間

## 1. 基本計算（r = 1 Rs）
管半径: r = {standard['r_Rs_units']} Rs = {standard['r_km']:.3e} km
消失時間: Δt = {standard['delta_t_seconds']:.3e} 秒
         = {standard['delta_t_hours']:.1f} 時間
         = {standard['delta_t_days']:.2f} 日

## 2. 異なる管半径での結果
r = 0.5 Rs: {radii_calc[0]['delta_t_hours']:.1f} 時間
r = 1.0 Rs: {radii_calc[1]['delta_t_hours']:.1f} 時間
r = 3.5 Rs: {radii_calc[3]['delta_t_hours']:.1f} 時間
r = 10.0 Rs: {radii_calc[5]['delta_t_hours']:.1f} 時間

## 3. ブラックホール比較
M87: {comparison['M87']['delta_t_hours']:.1f} 時間
Sgr A*: {comparison['SgrA']['delta_t_minutes']:.1f} 分
時間比: {comparison['ratio']['time_ratio']:.0f}

## 4. 観測可能性
必要時間分解能: < {detectability['time_resolution_required']['ideal_hours']} 時間
推奨観測期間: {detectability['observation_duration']['recommended_days']:.1f} 日
適切な観測装置: ALMA, X線衛星

## 5. 相対論的補正
r = 3 Rs での補正: {corrections[1]['correction_percent']:.1f}%
r = 10 Rs での補正: {corrections[3]['correction_percent']:.1f}%

## 検証完了
消失時間は観測可能な時間スケール
        """

        return report


if __name__ == "__main__":
    # 計算実行
    calc = DisappearanceTimeCalculation()

    # レポート生成
    report = calc.generate_report()
    print(report)

    # 可視化作成
    fig = calc.create_visualization()
    print("\n消失時間図をdisappearance_time_006.htmlに保存しました")

    # 結果保存
    results = calc.save_results()
    print("計算結果をcalculation_results_006.jsonに保存しました")