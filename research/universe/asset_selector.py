from __future__ import annotations

def validate_assets(assets: list[str]) -> list[str]:
    clean = []
    for asset in assets:
        asset = str(asset).strip()
        if asset and asset not in clean:
            clean.append(asset)
    if len(clean) > 10:
        raise ValueError("El universo máximo es de 10 activos.")
    return clean
