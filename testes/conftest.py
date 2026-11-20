"""Configuração comum dos testes do MARIA (pytest). Rodar da pasta principal:  python -m pytest testes -q"""
import ast
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "modelo"))
sys.argv = ["pytest"]


@pytest.fixture
def isolado(tmp_path, monkeypatch):
    """Usuários, auditoria e chave de pseudonimização em pasta temporária (não toca nos dados reais)."""
    import acesso, anonimizacao, auditoria
    monkeypatch.setattr(acesso, "ARQ_USUARIOS", tmp_path / "usuarios.json")
    monkeypatch.setattr(auditoria, "ARQ_AUDITORIA", tmp_path / "auditoria.jsonl")
    monkeypatch.setattr(anonimizacao, "ARQ_CHAVE", tmp_path / "chave.key")
    return tmp_path


def funcoes_do_dashboard(*nomes):
    """Extrai funções puras do maria_dashboard.py sem executar a interface."""
    import numpy as np, pandas as pd, re, io
    src = (RAIZ / "maria_dashboard.py").read_text(encoding="utf-8")
    ns = {"np": np, "pd": pd, "re": re, "io": io}
    for no in ast.parse(src).body:          # constantes do módulo (NOMES_EM_MAIUSCULAS) que as funções usam
        if isinstance(no, ast.Assign) and all(isinstance(t, (ast.Name, ast.Tuple)) for t in no.targets):
            alvos = [n.id for t in no.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if alvos and all(a.isupper() for a in alvos):
                try:
                    exec(compile(ast.Module([no], []), "dashboard", "exec"), ns)
                except Exception:
                    pass
    for no in ast.parse(src).body:
        if isinstance(no, ast.FunctionDef) and no.name in nomes:
            exec(compile(ast.Module([no], []), "dashboard", "exec"), ns)
    return ns


def precisa_da_release(*caminhos):
    """Pula o teste se faltar arquivo que vem no .zip da Release (pesos e exemplos DICOM; veja o README)."""
    import pytest
    faltando = [str(p) for p in caminhos if not Path(p).exists()]
    if faltando:
        pytest.skip("faltam arquivos da Release do GitHub (pesos/exemplos DICOM; veja o README): " + faltando[0])
