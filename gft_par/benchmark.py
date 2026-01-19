"""
Benchmark script para comparar performance entre versões do ROIFT
Uso: python benchmark.py
"""

import subprocess
import time
import os
from pathlib import Path


def run_oiftrelax(
    exe_path, volume, seeds, pol, niter, percentile, output, measure_time=True
):
    """Executa oiftrelax e retorna o tempo de execução"""
    cmd = [
        str(exe_path),
        str(volume),
        str(seeds),
        str(pol),
        str(niter),
        str(percentile),
        str(output),
    ]

    start = time.time()
    result = subprocess.run(cmd)  # Exibe debugs (sem capture_output)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Erro ao executar {exe_path} (return code: {result.returncode})")
        return None

    return elapsed


def compare_results(output1, output2):
    """Compara se dois volumes de saída são idênticos"""
    # Você pode usar SimpleITK ou nibabel aqui para comparar as imagens
    # Por simplicidade, vamos apenas verificar se os arquivos existem
    return os.path.exists(output1) and os.path.exists(output2)


def main():
    # Configurações
    volume = Path("nifti/test_volume.nii")
    seeds = Path("nifti/test_seeds.txt")
    pol = -0.5
    niter = 50
    percentile = 0

    # Executáveis (ajuste os caminhos conforme necessário)
    original_exe = Path("build/Debug/oiftrelax.exe")
    parallel_exe = Path("build/gft_par/Debug/oiftrelax_parallel.exe")

    # Verificar se arquivos existem
    if not volume.exists():
        print(f"Erro: {volume} não encontrado")
        return
    if not seeds.exists():
        print(f"Erro: {seeds} não encontrado")
        return

    print("=" * 60)
    print("BENCHMARK ROIFT - Original vs Paralelo (OpenMP)")
    print("=" * 60)
    print(f"Volume: {volume}")
    print(f"Seeds: {seeds}")
    print(f"Polaridade: {pol}")
    print(f"Iterações: {niter}")
    print(f"Percentil: {percentile}")
    print("-" * 60)

    # Teste com versão original
    if original_exe.exists():
        print("\n[1/2] Executando versão ORIGINAL (single-thread)...")
        output_orig = "result_original.nii.gz"
        time_orig = run_oiftrelax(
            original_exe, volume, seeds, pol, niter, percentile, output_orig
        )
        if time_orig:
            print(f"✓ Tempo: {time_orig:.2f}s")
        else:
            print("✗ Falhou")
    else:
        print(f"\n✗ Executável original não encontrado: {original_exe}")
        time_orig = None

    # Teste com versão paralela
    if parallel_exe.exists():
        # Testar com diferentes números de threads
        thread_counts = [2, 4, 8, 16]
        times_parallel = {}

        for num_threads in thread_counts:
            print(f"\n[2/2] Executando versão PARALELA ({num_threads} threads)...")
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            output_par = f"result_parallel_{num_threads}t.nii.gz"
            time_par = run_oiftrelax(
                parallel_exe, volume, seeds, pol, niter, percentile, output_par
            )
            if time_par:
                print(f"✓ Tempo: {time_par:.2f}s")
                times_parallel[num_threads] = time_par
            else:
                print("✗ Falhou")
    else:
        print(f"\n✗ Executável paralelo não encontrado: {parallel_exe}")
        times_parallel = {}

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)

    if time_orig:
        print(f"Original (1 thread):  {time_orig:.2f}s")

    if times_parallel:
        print("\nParalelo (OpenMP):")
        for threads, time_val in times_parallel.items():
            if time_orig:
                speedup = time_orig / time_val
                print(
                    f"  {threads} threads:  {time_val:.2f}s  (speedup: {speedup:.2f}x)"
                )
            else:
                print(f"  {threads} threads:  {time_val:.2f}s")

    # Comparar resultados
    if time_orig and times_parallel:
        print("\n" + "-" * 60)
        print("Verificando consistência dos resultados...")
        output_orig = "result_original.nii.gz"
        output_par = f"result_parallel_{list(times_parallel.keys())[0]}t.nii.gz"

        if compare_results(output_orig, output_par):
            print("✓ Ambas as versões geraram saídas")
            print("  (Use ITK-SNAP ou similar para comparar visualmente)")
        else:
            print("✗ Erro: Um ou ambos os arquivos de saída não foram gerados")

    print("\n" + "=" * 60)
    print("Benchmark concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
