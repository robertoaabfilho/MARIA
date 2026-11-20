"""
Controle de acesso do MARIA: usuários locais com senha (LGPD — só pessoas autorizadas acessam dados de saúde).

Arquivo: config/usuarios.json (fora do git). As senhas NUNCA são guardadas: só um hash scrypt com sal aleatório.
Perfis: "usuario" (classifica, revisa, anota) e "administrador" (também retreina, restaura versões, calibra,
importa/exporta o banco de anotações, administra usuários e vê a auditoria).

    python3 acesso.py --listar
    python3 acesso.py --adicionar joana --nome "Dra. Joana Silva" --perfil administrador   (pede a senha)
    python3 acesso.py --senha joana                                                        (troca a senha)
    python3 acesso.py --remover joana
"""
import sys
import argparse
import getpass
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ_USUARIOS = Path(os.environ.get("MARIA_USUARIOS") or RAIZ / "config" / "usuarios.json")   # variável: testes
PERFIS = ("usuario", "administrador")
SENHA_MINIMA = 8


def hash_senha(senha):
    sal = os.urandom(16)
    h = hashlib.scrypt(senha.encode("utf-8"), salt=sal, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${sal.hex()}${h.hex()}"


def conferir_senha(senha, armazenado):
    try:
        metodo, sal, h = armazenado.split("$")
        calc = hashlib.scrypt(senha.encode("utf-8"), salt=bytes.fromhex(sal), n=2 ** 14, r=8, p=1, dklen=32)
        return metodo == "scrypt" and hmac.compare_digest(calc.hex(), h)
    except Exception:
        return False


def carregar():
    if not ARQ_USUARIOS.is_file():
        return {}
    return json.loads(ARQ_USUARIOS.read_text(encoding="utf-8")).get("usuarios", {})


def salvar(usuarios):
    ARQ_USUARIOS.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARQ_USUARIOS.with_suffix(".tmp")
    tmp.write_text(json.dumps({"usuarios": usuarios}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ARQ_USUARIOS)
    try:
        os.chmod(ARQ_USUARIOS, 0o600)                 # só o dono lê (no Windows, vale a permissão da pasta)
    except OSError:
        pass


def validar_senha(senha):
    if len(senha) < SENHA_MINIMA:
        return f"a senha precisa ter pelo menos {SENHA_MINIMA} caracteres"
    if senha.isdigit() or senha.isalpha():
        return "use letras e números (ou símbolos) na senha"
    return None


def criar(login, nome, perfil, senha):
    login = login.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", login):
        raise ValueError("login: 3 a 32 caracteres (letras minúsculas, números, ponto, hífen ou sublinhado)")
    if perfil not in PERFIS:
        raise ValueError(f"perfil deve ser {' ou '.join(PERFIS)}")
    erro = validar_senha(senha)
    if erro:
        raise ValueError(erro)
    usuarios = carregar()
    if login in usuarios:
        raise ValueError(f"o login '{login}' já existe")
    usuarios[login] = {"nome": nome.strip() or login, "perfil": perfil, "senha": hash_senha(senha),
                       "criado_em": time.strftime("%Y-%m-%d %H:%M"), "ativo": True}
    salvar(usuarios)
    return usuarios[login]


def trocar_senha(login, senha):
    erro = validar_senha(senha)
    if erro:
        raise ValueError(erro)
    usuarios = carregar()
    usuarios[login]["senha"] = hash_senha(senha)
    usuarios[login]["senha_trocada_em"] = time.strftime("%Y-%m-%d %H:%M")
    salvar(usuarios)


def remover(login):
    usuarios = carregar()
    if login in usuarios and usuarios[login]["perfil"] == "administrador" and \
            sum(u["perfil"] == "administrador" for u in usuarios.values()) == 1:
        raise ValueError("não é possível remover o único administrador")
    usuarios.pop(login, None)
    salvar(usuarios)


def autenticar(login, senha):
    """-> dados do usuário (sem o hash) ou None."""
    u = carregar().get((login or "").strip().lower())
    if not u or not u.get("ativo", True) or not conferir_senha(senha, u["senha"]):
        return None
    return {"login": login.strip().lower(), "nome": u["nome"], "perfil": u["perfil"]}


if __name__ == "__main__":
    for _fluxo in (sys.stdout, sys.stderr):          # Windows: saída redirecionada usa cp1252 e quebra com ✅/❌
        try:
            _fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--adicionar")
    ap.add_argument("--nome", default="")
    ap.add_argument("--perfil", default="usuario", choices=PERFIS)
    ap.add_argument("--senha")
    ap.add_argument("--remover")
    a = ap.parse_args()
    if a.adicionar:
        s1 = getpass.getpass("Senha: "); s2 = getpass.getpass("Repita a senha: ")
        if s1 != s2:
            raise SystemExit("❌ as senhas não conferem")
        criar(a.adicionar, a.nome, a.perfil, s1); print(f"✅ usuário '{a.adicionar}' criado ({a.perfil})")
    elif a.senha:
        s1 = getpass.getpass("Nova senha: "); s2 = getpass.getpass("Repita a nova senha: ")
        if s1 != s2:
            raise SystemExit("❌ as senhas não conferem")
        trocar_senha(a.senha, s1); print("✅ senha trocada")
    elif a.remover:
        remover(a.remover); print(f"✅ usuário '{a.remover}' removido")
    else:
        for login, u in carregar().items():
            print(f"{login:<20} {u['perfil']:<14} {u['nome']}")
