from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


TARGET = Path("src/features/build_fighter_snapshots.py")


def replace_exact(
    text: str,
    old: str,
    new: str,
    label: str,
    expected_count: int = 1,
) -> str:
    count = text.count(old)

    if count != expected_count:
        raise RuntimeError(
            f"No se pudo aplicar '{label}': "
            f"se esperaban {expected_count} coincidencias "
            f"y se encontraron {count}."
        )

    return text.replace(old, new)


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"No se encuentra {TARGET.resolve()}. "
            "Ejecuta este parche desde la raíz UFC_PREDICTOR."
        )

    text = TARGET.read_text(encoding="utf-8")

    # Permite ejecutar el parche más de una vez sin romper el archivo.
    already_patched = (
        "fights_with_control_stats: int = 0" in text
        and '"control_stats_coverage_before"' in text
        and "if pd.notna(own_stats.control_seconds):" in text
        and '"control_seconds",\n    ]\n\n    null_counts' not in text
    )

    if already_patched:
        print("El archivo ya contiene la corrección de control_seconds.")
        print(f"Archivo: {TARGET.resolve()}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(
        f"{TARGET.stem}_before_control_patch_{timestamp}{TARGET.suffix}"
    )
    shutil.copy2(TARGET, backup)

    text = replace_exact(
        text,
        '''    # Cobertura de estadísticas detalladas.
    fights_with_stats: int = 0
    stats_fight_time_seconds: int = 0

    # Striking.
''',
        '''    # Cobertura de estadísticas detalladas.
    fights_with_stats: int = 0
    stats_fight_time_seconds: int = 0

    # El tiempo de control no está disponible en todas las filas
    # históricas, aunque sí existan las demás estadísticas.
    fights_with_control_stats: int = 0

    # Striking.
''',
        "estado de cobertura de control",
    )

    text = replace_exact(
        text,
        '''    stats_coverage = safe_ratio(
        state.fights_with_stats,
        state.fights,
    )

    return {
''',
        '''    stats_coverage = safe_ratio(
        state.fights_with_stats,
        state.fights,
    )

    control_stats_coverage = safe_ratio(
        state.fights_with_control_stats,
        state.fights,
    )

    return {
''',
        "cálculo de cobertura de control",
    )

    text = replace_exact(
        text,
        '''        "stats_coverage_before": stats_coverage,

        # Striking acumulado previo.
''',
        '''        "stats_coverage_before": stats_coverage,
        "fights_with_control_stats_before": (
            state.fights_with_control_stats
        ),
        "control_stats_coverage_before": (
            control_stats_coverage
        ),

        # Striking acumulado previo.
''',
        "columnas de cobertura de control",
    )

    text = replace_exact(
        text,
        '''    # Grappling y control.
    state.control_seconds += int(own_stats.control_seconds)
    state.knockdowns += int(own_stats.knockdowns)
''',
        '''    # El tiempo de control es parcialmente ausente en UFCStats.
    # Un valor nulo significa desconocido, no cero.
    if pd.notna(own_stats.control_seconds):
        state.fights_with_control_stats += 1
        state.control_seconds += int(own_stats.control_seconds)

    # Resto de métricas de grappling.
    state.knockdowns += int(own_stats.knockdowns)
''',
        "actualización nullable de control_seconds",
    )

    text = replace_exact(
        text,
        '''    non_negative_columns = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_seconds",
    ]
''',
        '''    non_negative_columns = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
    ]
''',
        "exclusión de control_seconds de campos obligatorios",
    )

    text = replace_exact(
        text,
        '''    if negative_columns:
        raise ValueError(
            "Hay estadísticas negativas en estas columnas: "
            f"{negative_columns}"
        )

    invalid_sig_str = (
''',
        '''    if negative_columns:
        raise ValueError(
            "Hay estadísticas negativas en estas columnas: "
            f"{negative_columns}"
        )

    # control_seconds admite nulos, pero no valores negativos.
    invalid_control = (
        fight_totals["control_seconds"]
        .dropna()
        .lt(0)
    )

    if invalid_control.any():
        raise ValueError(
            "Hay valores negativos en control_seconds."
        )

    invalid_sig_str = (
''',
        "validación independiente de control_seconds",
    )

    text = replace_exact(
        text,
        '''    if invalid_stats_counts.any():
        raise ValueError(
            "fights_with_stats_before supera fights_before."
        )

    negative_days = (
''',
        '''    if invalid_stats_counts.any():
        raise ValueError(
            "fights_with_stats_before supera fights_before."
        )

    invalid_control_counts = (
        snapshots["fights_with_control_stats_before"]
        > snapshots["fights_with_stats_before"]
    )

    if invalid_control_counts.any():
        raise ValueError(
            "fights_with_control_stats_before supera "
            "fights_with_stats_before."
        )

    negative_days = (
''',
        "validación de cobertura de control",
    )

    # Este bloque aparece dos veces:
    # 1) validación del primer snapshot
    # 2) lista de columnas enteras
    text = replace_exact(
        text,
        '''        "fights_with_stats_before",
        "stats_fight_time_seconds_before",
        "sig_str_landed_before",
''',
        '''        "fights_with_stats_before",
        "stats_fight_time_seconds_before",
        "fights_with_control_stats_before",
        "sig_str_landed_before",
''',
        "incorporación de fights_with_control_stats_before",
        expected_count=2,
    )

    text = replace_exact(
        text,
        '''        "stats_coverage_before",
        "sig_str_landed_per_min_before",
''',
        '''        "stats_coverage_before",
        "control_stats_coverage_before",
        "sig_str_landed_per_min_before",
''',
        "tipo Float64 de control_stats_coverage_before",
    )

    text = replace_exact(
        text,
        '''    fights_without_stats = (
        len(fights) - fights_with_stats
    )

    print()
''',
        '''    fights_without_stats = (
        len(fights) - fights_with_stats
    )

    missing_control_rows = int(
        fight_totals["control_seconds"].isna().sum()
    )
    missing_control_fights = int(
        fight_totals.loc[
            fight_totals["control_seconds"].isna(),
            "fight_id",
        ].nunique()
    )

    print()
''',
        "resumen de valores de control ausentes",
    )

    text = replace_exact(
        text,
        '''    print(
        "Combates sin estadísticas: "
        f"{fights_without_stats:,}"
    )
    print(
        "Peleadores con varios combates en un mismo evento: "
''',
        '''    print(
        "Combates sin estadísticas: "
        f"{fights_without_stats:,}"
    )
    print(
        "Filas sin tiempo de control: "
        f"{missing_control_rows:,}"
    )
    print(
        "Combates afectados por control ausente: "
        f"{missing_control_fights:,}"
    )
    print(
        "Peleadores con varios combates en un mismo evento: "
''',
        "impresión del resumen de control",
    )

    TARGET.write_text(text, encoding="utf-8")

    print("Parche aplicado correctamente.")
    print(f"Archivo modificado: {TARGET.resolve()}")
    print(f"Copia de seguridad: {backup.resolve()}")
    print(f"Líneas actuales: {len(text.splitlines())}")
    print()
    print("Comprobaciones:")
    print(
        "  fights_with_control_stats:",
        "fights_with_control_stats: int = 0" in text,
    )
    print(
        "  control nullable:",
        "if pd.notna(own_stats.control_seconds):" in text,
    )
    print(
        "  cobertura de control:",
        '"control_stats_coverage_before"' in text,
    )


if __name__ == "__main__":
    main()