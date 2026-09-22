# Despliegue — GitHub privado + servidor

## Fuente de datos: exclusivamente MetaTrader 5

CapitalQuant obtiene datos de mercado ÚNICAMENTE desde MetaTrader 5. No hay
ninguna fuente alternativa (no se usa Yahoo Finance ni ningún otro
proveedor externo). Esto requiere que el servidor donde corre la app tenga
el paquete `MetaTrader5` instalado y un terminal MT5 abierto/logueado
(Windows, o una VM/contenedor Windows) — el paquete `MetaTrader5` solo
tiene wheel oficial para Windows.

Esto significa que en un servidor Linux sin terminal MT5, la app arrancará
sin conexión y solo podrá servir lo que ya exista en el caché SQLite local
(`data/capitalquant_cache.db`) — no habrá datos frescos hasta que se
conecte a un MT5 real. Si necesitas que la nube tenga datos desde el
primer momento, sube ese `.db` ya poblado al repo (o a un volumen
persistente), o despliega directamente en un servidor Windows con MT5.

**Ejecución real de órdenes** (página "Trading Algorítmico", vía
`execution/mt5_execution.py`) también requiere sí o sí un terminal MT5 real
conectado — no existe ningún modo alternativo de envío de órdenes.

## 1. Subir el repositorio a GitHub (privado)

Este proyecto ya está preparado como repositorio Git local (`git init` + commit
inicial). Para subirlo:

```bash
# 1. Crea el repo vacío en GitHub (sin README/licencia, para no chocar con el commit local)
#    Web: https://github.com/new  -> marca "Private"
#    o con GitHub CLI:
gh repo create TU_USUARIO/capitalquant --private --source=. --remote=origin

# 2. Si lo creaste desde la web en vez de con gh:
git remote add origin git@github.com:TU_USUARIO/capitalquant.git

# 3. Sube el código
git branch -M main
git push -u origin main
```

Usa SSH (`git@github.com:...`) o un Personal Access Token si usas HTTPS —
GitHub ya no acepta usuario/contraseña simple para `git push`.

## 2. Clonar en el servidor

```bash
git clone git@github.com:TU_USUARIO/capitalquant.git
cd capitalquant
```

Si el servidor no tiene una clave SSH asociada a tu cuenta de GitHub, genera
una (`ssh-keygen -t ed25519`) y añádela en GitHub → Settings → SSH and GPG keys,
o clona por HTTPS con un token.

## 3. Entorno Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota sobre MetaTrader5:** el paquete `MetaTrader5` solo tiene wheel para
Windows (`sys_platform == "win32"` en `requirements.txt`), así que en un
servidor Linux se instala todo *menos* ese paquete. En ese caso
`market_data/mt5_provider.py` detecta la ausencia, no rompe la app, pero
tampoco hay ninguna fuente de datos alternativa: la plataforma solo sirve
el caché local hasta que corra en una máquina con MT5 real disponible.

## 4. Ejecutar la app

```bash
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

Para producción, ejecútalo detrás de un proceso persistente:

```bash
# systemd (ejemplo: /etc/systemd/system/capitalquant.service)
[Unit]
Description=CapitalQuant Streamlit
After=network.target

[Service]
User=capitalquant
WorkingDirectory=/opt/capitalquant
ExecStart=/opt/capitalquant/venv/bin/streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now capitalquant
```

Y opcionalmente un reverse proxy (nginx/caddy) delante para TLS.

## 5. Actualizar el servidor tras nuevos cambios

```bash
cd capitalquant
git pull
source venv/bin/activate
pip install -r requirements.txt   # solo si cambiaron dependencias
sudo systemctl restart capitalquant
```

## 6. Qué NO se sube al repo (ver `.gitignore`)

- `venv/` — entorno virtual
- `data/*.db` y `data/*/daily|1h|4h|15m|5m/` — caché de datos de mercado,
  se regenera en cada máquina
- `reports/optimizations/*.json` — resultados de corridas locales
- `.pytest_cache/`, `__pycache__/`

Estos se recrean solos la primera vez que se usa la app en el servidor
(`config/settings.py` crea los directorios necesarios).
