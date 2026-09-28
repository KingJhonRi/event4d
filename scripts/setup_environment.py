"""Bootstrap this Windows CPU research demo, including SOCKS-proxy hosts."""
from __future__ import annotations
import json,subprocess,sys,urllib.request,venv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=ROOT/".venv"
if not (env/"Scripts/python.exe").exists(): venv.create(env,with_pip=True)
python=env/"Scripts/python.exe"

def pip(*args): subprocess.run([str(python),"-m","pip",*args],check=True)

try:
    pip("install","-r",str(ROOT/"requirements.txt"))
except subprocess.CalledProcessError:
    # pip cannot use a SOCKS proxy until PySocks has been installed.
    info=json.load(urllib.request.urlopen("https://pypi.org/pypi/PySocks/json"))
    wheel=next(x for x in info["urls"] if x["filename"].endswith("py3-none-any.whl"))
    local=ROOT/"models"/wheel["filename"]
    local.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(wheel["url"],local)
    pip("install","--no-index",str(local))
    pip("install","-r",str(ROOT/"requirements.txt"))

repo=ROOT/"third_party/GroundingDINO"
if not repo.exists(): subprocess.run(["git","clone","--depth","1","https://github.com/IDEA-Research/GroundingDINO.git",str(repo)],check=True)
log=ROOT/"logs/install_groundingdino.log"
log.parent.mkdir(parents=True,exist_ok=True)
with log.open("w") as f:
    result=subprocess.run([str(python),"-m","pip","install","-e",str(repo),"--no-build-isolation"],stdout=f,stderr=subprocess.STDOUT)
if result.returncode:
    print(f"Official GroundingDINO install failed; see {log}. Transformers fallback remains available.")
else:
    print("Official GroundingDINO installed.")
