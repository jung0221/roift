#!/usr/bin/env python3
"""
Debug Timing Comparison Tool for ROIFT

Executa as versões original e paralela, extrai os tempos de debug
e cria um relatório comparativo de performance.

Uso:
    python3 debug_compare.py <volume.nii.gz> <seeds.txt> [output_dir]
"""

import subprocess
import re
import sys
import os
from pathlib import Path
import json
from datetime import datetime


class TimingExtractor:
    """Extrai tempos de debug da saída do oiftrelax"""

    def __init__(self, executable, volume, seeds, pol=0.5, niter=50, percentile=50):
        self.executable = executable
        self.volume = volume
        self.seeds = seeds
        self.pol = pol
        self.niter = niter
        self.percentile = percentile
        self.timings = {}
        self.total_time = 0.0

    def run(self):
        """Executa o comando e captura output"""
        cmd = [
            self.executable,
            self.volume,
            self.seeds,
            str(self.pol),
            str(self.niter),
            str(self.percentile),
            "debug_output.nii.gz",  # output file (not used in this test)
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minutes timeout
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            print(f"ERROR: {self.executable} timed out after 5 minutes")
            return None
        except FileNotFoundError:
            print(f"ERROR: {self.executable} not found")
            return None

    def parse_timing_summary(self, output):
        """Extrai o sumário de timing da saída"""
        if not output:
            return False

        # Padrão: "Event Name   Total (ms)  Calls  Avg (ms)"
        pattern = r"([^=\n]+?)\s+(\d+\.?\d*)\s+ms"

        # Procura por "[DEBUG] <<< END: <name>"
        end_pattern = r"\[DEBUG\] <<< END: ([^|]+)\s*\|\s*Elapsed:\s*([\d.]+)\s*ms"

        matches = re.findall(end_pattern, output)
        if not matches:
            return False

        self.timings = {}
        for name, elapsed in matches:
            name = name.strip()
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(float(elapsed))

        # Calcular tempo total
        self.total_time = sum(sum(times) for times in self.timings.values())

        return len(self.timings) > 0

    def get_summary(self):
        """Retorna sumário formatado"""
        if not self.timings:
            return None

        summary = {
            "total_ms": self.total_time,
            "total_sec": self.total_time / 1000.0,
            "events": {},
        }

        for name, times in self.timings.items():
            summary["events"][name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }

        return summary


def format_timing_table(summary):
    """Formata sumário em tabela legível"""
    if not summary:
        return "Nenhum timing encontrado"

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(
        f"TIMING SUMMARY | Total: {summary['total_sec']:.2f}s ({summary['total_ms']:.0f}ms)"
    )
    lines.append("=" * 90)
    lines.append(f"{'Event':<50} {'Count':<8} {'Total (ms)':<15} {'Avg (ms)':<12}")
    lines.append("-" * 90)

    for name in sorted(summary["events"].keys()):
        evt = summary["events"][name]
        lines.append(
            f"{name:<50} {evt['count']:<8} {evt['total_ms']:<15.2f} {evt['avg_ms']:<12.2f}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def compare_timings(original_summary, parallel_summary):
    """Compara os dois sumários e calcula speedup"""
    if not original_summary or not parallel_summary:
        return None

    comparison = {
        "original_total_s": original_summary["total_sec"],
        "parallel_total_s": parallel_summary["total_sec"],
        "speedup": original_summary["total_ms"] / parallel_summary["total_ms"],
        "time_saved_s": (original_summary["total_ms"] - parallel_summary["total_ms"])
        / 1000.0,
        "events": {},
    }

    # Comparar eventos individuais
    for name in original_summary["events"]:
        if name in parallel_summary["events"]:
            orig = original_summary["events"][name]
            para = parallel_summary["events"][name]

            if para["total_ms"] > 0:
                speedup = orig["total_ms"] / para["total_ms"]
            else:
                speedup = 1.0

            comparison["events"][name] = {
                "original_ms": orig["total_ms"],
                "parallel_ms": para["total_ms"],
                "speedup": speedup,
                "original_avg_ms": orig["avg_ms"],
                "parallel_avg_ms": para["avg_ms"],
            }

    return comparison


def format_comparison(comparison):
    """Formata comparação em tabela legível"""
    if not comparison:
        return "Não foi possível comparar"

    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("PERFORMANCE COMPARISON: Original vs Parallel")
    lines.append("=" * 110)

    lines.append(f"\nTEMPO TOTAL:")
    lines.append(
        f"  Original (sequencial):  {comparison['original_total_s']:.2f}s ({comparison['original_total_s']*1000:.0f}ms)"
    )
    lines.append(
        f"  Paralelo (OpenMP):      {comparison['parallel_total_s']:.2f}s ({comparison['parallel_total_s']*1000:.0f}ms)"
    )
    lines.append(f"  Speedup:                {comparison['speedup']:.2f}x")
    lines.append(f"  Tempo economizado:      {comparison['time_saved_s']:.2f}s")

    lines.append(f"\n{'Event':<50} {'Original':<15} {'Paralelo':<15} {'Speedup':<10}")
    lines.append("-" * 110)

    for name in sorted(comparison["events"].keys()):
        evt = comparison["events"][name]
        lines.append(
            f"{name:<50} {evt['original_ms']:<15.2f} {evt['parallel_ms']:<15.2f} {evt['speedup']:<10.2f}x"
        )

    lines.append("=" * 110)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 debug_compare.py <volume.nii.gz> <seeds.txt> [output_dir]")
        print("\nExemplo:")
        print("  python3 debug_compare.py volume.nii.gz seeds.txt ./results")
        sys.exit(1)

    volume = sys.argv[1]
    seeds = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."

    # Verificar arquivos
    if not os.path.exists(volume):
        print(f"ERRO: Arquivo de volume não encontrado: {volume}")
        sys.exit(1)
    if not os.path.exists(seeds):
        print(f"ERRO: Arquivo de seeds não encontrado: {seeds}")
        sys.exit(1)

    # Criar diretório de output se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Encontrar executáveis
    build_dir = Path(__file__).parent.parent / "build"

    # Tentar encontrar os executáveis
    possible_paths = [
        build_dir / "Release" / "oiftrelax.exe",
        build_dir / "Debug" / "oiftrelax.exe",
        build_dir / "oiftrelax.exe",
        Path("./oiftrelax.exe"),
        Path("oiftrelax"),
    ]

    original_exe = None
    for path in possible_paths:
        if path.exists():
            original_exe = str(path)
            break

    if not original_exe:
        print("ERRO: Não foi possível encontrar oiftrelax")
        print(f"Procurei em: {[str(p) for p in possible_paths]}")
        sys.exit(1)

    # Buscar versão paralela
    parallel_exe = None
    possible_parallel = [
        build_dir / "Release" / "oiftrelax_parallel.exe",
        build_dir / "Debug" / "oiftrelax_parallel.exe",
        build_dir / "oiftrelax_parallel.exe",
        Path("./oiftrelax_parallel.exe"),
        Path("oiftrelax_parallel"),
    ]

    for path in possible_parallel:
        if path.exists():
            parallel_exe = str(path)
            break

    if not parallel_exe:
        print("AVISO: Versão paralela (oiftrelax_parallel) não encontrada")
        print("Apenas executarei a versão original")
        print()

    print(f"Volume: {volume}")
    print(f"Seeds: {seeds}")
    print(f"Original: {original_exe}")
    if parallel_exe:
        print(f"Paralelo: {parallel_exe}")
    print()

    # Executar versão original
    print("[1/2] Executando versão original...")
    extractor_orig = TimingExtractor(original_exe, volume, seeds)
    output_orig = extractor_orig.run()

    if output_orig and extractor_orig.parse_timing_summary(output_orig):
        print("✓ Versão original completada com sucesso")
        summary_orig = extractor_orig.get_summary()
    else:
        print("✗ Falha ao executar versão original")
        summary_orig = None

    # Executar versão paralela (se encontrada)
    summary_para = None
    if parallel_exe:
        print("[2/2] Executando versão paralela...")
        extractor_para = TimingExtractor(parallel_exe, volume, seeds)
        output_para = extractor_para.run()

        if output_para and extractor_para.parse_timing_summary(output_para):
            print("✓ Versão paralela completada com sucesso")
            summary_para = extractor_para.get_summary()
        else:
            print("✗ Falha ao executar versão paralela")

    # Gerar relatórios
    report_lines = []
    report_lines.append(f"ROIFT Debug Timing Report - {datetime.now().isoformat()}")
    report_lines.append(f"Volume: {volume}")
    report_lines.append(f"Seeds: {seeds}")
    report_lines.append("")

    if summary_orig:
        print(format_timing_table(summary_orig))
        report_lines.append(format_timing_table(summary_orig))

    if summary_para:
        print(format_timing_table(summary_para))
        report_lines.append(format_timing_table(summary_para))

        comparison = compare_timings(summary_orig, summary_para)
        if comparison:
            print(format_comparison(comparison))
            report_lines.append(format_comparison(comparison))

            # Salvar JSON para análise posterior
            json_file = Path(output_dir) / "timing_comparison.json"
            with open(json_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "original": summary_orig,
                        "parallel": summary_para,
                        "comparison": comparison,
                    },
                    f,
                    indent=2,
                )
            print(f"\nJSON salvo em: {json_file}")

    # Salvar relatório de texto
    report_file = Path(output_dir) / "timing_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nRelatório salvo em: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
