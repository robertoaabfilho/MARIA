"""
Desempenho LOCAL do MARIA com os casos confirmados do serviço (banco de anotações): rótulos por lesão e por MAMA,
intervalos de confiança (bootstrap por mama), prontidão para calibrar, fila de casos para anotar e monitoramento
no tempo. Só numpy/pandas (roda em qualquer máquina, sem scikit-learn). Usado pela página "Anotações e retreino".
"""
import numpy as np
import pandas as pd

EXAMES_VALIDOS = ["Biópsia (anatomopatológico)", "Seguimento (estável por ≥ 2 anos)", "Outro exame padrão-ouro"]
MAMA_CONFIRMADA = ["Maligna", "Benigna", "Normal"]
MINIMO = {"analise": 10, "calibrar_total": 50, "calibrar_por_classe": 15, "estavel_total": 30, "estavel_por_classe": 10}
EPS = 1e-4


# ---------------------------------------------------------------- métricas
def auc(y, p):
    """AUC pela estatística de Mann-Whitney (empates = meio ponto)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    todos = np.concatenate([pos, neg])
    ordem = todos.argsort(kind="mergesort")
    postos = np.empty(len(todos))
    postos[ordem] = np.arange(1, len(todos) + 1)
    _, inv, cont = np.unique(todos, return_inverse=True, return_counts=True)
    soma = np.bincount(inv, weights=postos)
    postos = (soma / cont)[inv]
    return float((postos[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sens(y, p, limiar):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    return float((p[y == 1] >= limiar).mean()) if (y == 1).any() else np.nan


def espec(y, p, limiar):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    return float((p[y == 0] < limiar).mean()) if (y == 0).any() else np.nan


def ic_bootstrap(fn, y, p, grupos=None, n=1000, semente=0):
    """(valor, IC 2,5%, IC 97,5%) de fn(y, p) com bootstrap por grupo (mama/paciente: lesões da mesma mama juntas)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    valor = fn(y, p)
    if len(y) < 2:
        return valor, np.nan, np.nan
    grupos = np.arange(len(y)) if grupos is None else np.asarray(grupos)
    uni, inv = np.unique(grupos, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uni))]
    rng, vals = np.random.default_rng(semente), []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(uni), len(uni))])
        v = fn(y[s], p[s])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n * 0.5:
        return valor, np.nan, np.nan
    return valor, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def estabilidade(y):
    """'ok' | 'instável' (poucos casos para decidir) | 'insuficiente' (não dá para medir)."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if len(y) < MINIMO["analise"] or min(n1, n0) == 0:
        return "insuficiente"
    if len(y) < MINIMO["estavel_total"] or min(n1, n0) < MINIMO["estavel_por_classe"]:
        return "instável"
    return "ok"


def pode_calibrar(y):
    y = np.asarray(y).astype(int)
    return len(y) >= MINIMO["calibrar_total"] and min(int(y.sum()), int((y == 0).sum())) >= MINIMO["calibrar_por_classe"]


# ---------------------------------------------------------------- rótulos
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def lesoes_confirmadas(banco):
    """Lesões com patologia confirmada por exame válido (falso positivo fora) -> y, p (P bruta), grupo = mama."""
    b = banco[banco["patologia_confirmada"].isin(["Benigno", "Maligno"])
              & banco["exame_confirmacao"].isin(EXAMES_VALIDOS) & (banco["lesao_real"] != "não")].copy()
    b["p"] = _num(b["p_maligno"])
    b = b.dropna(subset=["p"])
    b["y"] = (b["patologia_confirmada"] == "Maligno").astype(int)
    b["grupo"] = b["rodada"] + "|" + b["mama"]
    return b


def mamas_confirmadas(banco):
    """Uma linha por mama (rodada + mama) com rótulo confirmado:
       maligna = alguma lesão maligna confirmada (exame válido) OU 'mama confirmada' = Maligna (exame válido);
       não maligna = 'mama confirmada' = Benigna ou Normal (exame válido; normal = seguimento) e nada maligno.
    Colunas: grupo, y, normal, escore (escore da mama do híbrido, se houver), p_maria (maior P bruta das lesões),
    prioridade, fabricante, idade, densidade, data, versao_modelo."""
    if banco.empty:
        return pd.DataFrame(columns=["grupo", "y", "normal", "escore", "p_maria"])
    b = banco.copy()
    b["grupo"] = b["rodada"] + "|" + b["mama"]
    valido = b["exame_confirmacao"].isin(EXAMES_VALIDOS)
    les_mal = (b["patologia_confirmada"] == "Maligno") & valido & (b["lesao_real"] != "não")
    linhas = []
    for g, d in b.groupby("grupo"):
        v = valido[d.index]
        mc = d.loc[v, "mama_confirmada"] if "mama_confirmada" in d else pd.Series(dtype=str)
        maligna = bool(les_mal[d.index].any() or (mc == "Maligna").any())
        negativa = bool((mc.isin(["Benigna", "Normal"])).any())
        if not maligna and not negativa:
            continue
        p_les = _num(d.loc[d["origem_lesao"].isin(["marcada pelo usuário", "detectada pelo modelo"]), "p_maligno"])
        esc = _num(d.get("escore_mama", pd.Series(dtype=str))).dropna()
        linhas.append({"grupo": g, "y": int(maligna), "normal": bool(not maligna and (mc == "Normal").any()),
                       "escore": float(esc.iloc[0]) if len(esc) else np.nan,
                       "p_maria": float(p_les.max()) if p_les.notna().any() else np.nan,
                       "prioridade": d["prioridade"].iloc[0], "fabricante": d["fabricante"].replace("", np.nan).dropna()
                       .iloc[0] if (d["fabricante"] != "").any() else "não informado",
                       "idade": _num(d["idade"]).dropna().iloc[0] if _num(d["idade"]).notna().any() else np.nan,
                       "densidade": d["densidade_confirmada"].replace("", np.nan).dropna().iloc[0]
                       if (d["densidade_confirmada"] != "").any() else d["densidade_modelo"].iloc[0],
                       "data": d["data"].iloc[0], "versao_modelo": d["versao_modelo"].iloc[0]})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- monitoramento
def por_periodo(df, coluna_score, limiar, freq="M"):
    """Desempenho por mês (ou outra frequência): n, malignos, AUC e sensibilidade com IC, especificidade."""
    if df.empty:
        return pd.DataFrame()
    d = df.dropna(subset=[coluna_score]).copy()
    d["periodo"] = pd.to_datetime(d["data"], errors="coerce").dt.to_period(freq).astype(str)
    out = []
    for per, g in d.groupby("periodo"):
        y, p, gr = g["y"].to_numpy(), g[coluna_score].to_numpy(), g["grupo"].to_numpy()
        a = ic_bootstrap(auc, y, p, gr, n=500)
        s = ic_bootstrap(lambda yy, pp: sens(yy, pp, limiar), y, p, gr, n=500)
        out.append({"periodo": per, "casos": len(g), "malignos": int(y.sum()), "AUC": a[0], "AUC_inf": a[1],
                    "AUC_sup": a[2], "sensibilidade": s[0], "sens_inf": s[1], "sens_sup": s[2],
                    "especificidade": espec(y, p, limiar)})
    return pd.DataFrame(out)


def alertas(tabela, sens_referencia, minimo_casos=20):
    """Períodos (com casos suficientes) em que o IC 95% da sensibilidade fica todo ABAIXO da referência."""
    if tabela.empty:
        return []
    t = tabela[(tabela["casos"] >= minimo_casos) & tabela["sens_sup"].notna()]
    return [f"{r.periodo}: sensibilidade {r.sensibilidade:.0%} (IC 95% até {r.sens_sup:.0%}) abaixo da referência "
            f"{sens_referencia:.0%}" for r in t.itertuples() if r.sens_sup < sens_referencia]


# ---------------------------------------------------------------- fila de casos para anotar
MOTIVOS = {
    "discordancia": "GMIC e MARIA discordam",
    "alta_sem_lesao": "mama em prioridade alta/média sem lesão detectada",
    "incerto": "resultado incerto (redes discordam ou P intermediária)",
    "equipamento_raro": "equipamento com poucos casos confirmados",
    "sem_confirmacao": "prioridade alta ainda sem confirmação",
}


def _z(p, media, desvio, eps=1e-3):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return (np.log(p / (1 - p)) - media) / desvio


def fila_para_anotar(resultados, ja_anotados, fabricantes_banco, limite=50, cfg_hibrido=None, discordancia_z=2.0):
    """resultados: DataFrame com as linhas de results.csv de várias rodadas (+ coluna 'rodada').
    ja_anotados: conjunto de amostras (originais) já no banco. Devolve as linhas mais úteis para revisar,
    com pontos e motivos (uma pontuação simples: cada motivo vale pontos; a prioridade alta desempata)."""
    if resultados.empty:
        return resultados
    r = resultados[~resultados["amostra"].isin(ja_anotados)].copy()
    if r.empty:
        return r
    for c in ("p_gmic_mama", "escore_mama", "p_maligno", "p_maligno_calibrada", "incerteza"):
        r[c] = _num(r[c]) if c in r else np.nan
    motivos, pontos = [[] for _ in range(len(r))], np.zeros(len(r))
    # discordância GMIC × MARIA: as duas P na MESMA escala do híbrido (logits padronizados pelo CMMD) muito
    # diferentes (>= discordancia_z desvios); sem a configuração do híbrido, não há como comparar as escalas
    if r["p_gmic_mama"].notna().any() and cfg_hibrido:
        c = cfg_hibrido
        pm = r.groupby("mama")["p_maligno"].transform("max")
        disc = pd.Series(np.abs(_z(r["p_gmic_mama"], c["gmic_media"], c["gmic_desvio"])
                                - _z(pm.fillna(c["maria_sem_lesao"]), c["maria_media"], c["maria_desvio"]))
                         >= discordancia_z, index=r.index) & r["p_gmic_mama"].notna()
        for i in np.where(disc.fillna(False).to_numpy())[0]:
            motivos[i].append(MOTIVOS["discordancia"]); pontos[i] += 3
    sem_lesao = (r.get("origem_lesao", "") == "nenhuma lesão encontrada") & r.get("prioridade", "").isin(["alta", "média"])
    for i in np.where(np.asarray(sem_lesao))[0]:
        motivos[i].append(MOTIVOS["alta_sem_lesao"]); pontos[i] += 3
    incerto = (r["incerteza"] > 0.15) | r["p_maligno_calibrada"].between(0.35, 0.65)
    for i in np.where(incerto.fillna(False).to_numpy())[0]:
        motivos[i].append(MOTIVOS["incerto"]); pontos[i] += 2
    fab = r.get("fabricante", pd.Series([""] * len(r), index=r.index)).fillna("").astype(str)
    raro = fab.ne("") & ~fab.isin({f for f, n in fabricantes_banco.items() if n >= 10})
    for i in np.where(raro.to_numpy())[0]:
        motivos[i].append(MOTIVOS["equipamento_raro"]); pontos[i] += 1
    alta = (r.get("prioridade", "") == "alta").to_numpy()
    for i in np.where(alta)[0]:
        if not motivos[i]:
            motivos[i].append(MOTIVOS["sem_confirmacao"])
        pontos[i] += 1
    r["pontos"], r["motivos"] = pontos, ["; ".join(m) for m in motivos]
    r = r[r["pontos"] > 0].sort_values(["pontos", "escore_mama", "p_maligno_calibrada"], ascending=False)
    return r.head(limite)


# ---------------------------------------------------------------- modo sombra: em uso × candidato
DENS_NUM = {"1": 1, "2": 2, "3": 3, "4": 4, "A": 1, "B": 2, "C": 3, "D": 4}


def comparar_sombra(banco, modelo, carimbo):
    """Compara o modelo em uso com o candidato (modo sombra) nos casos confirmados classificados com ele.
    -> DataFrame (medida, em uso, candidato, casos)."""
    if banco.empty or "versao_sombra" not in banco:
        return pd.DataFrame()
    b = banco[banco["versao_sombra"].str.contains(f"{modelo}:{carimbo}", regex=False)]
    linhas = []
    if modelo == "densidade":
        d = b[(b["densidade_confirmada"] != "") & (b["densidade_origem"] == "inferida pelo modelo")
              & (b["densidade_sombra"] != "")].drop_duplicates(["rodada", "mama"])
        if len(d):
            v = d["densidade_confirmada"].map(DENS_NUM)
            a, c = _num(d["densidade_modelo"]), _num(d["densidade_sombra"])
            linhas += [{"medida": "densidade exata (acerto)", "em uso": float((a == v).mean()),
                        "candidato": float((c == v).mean()), "casos": len(d)},
                       {"medida": "densa × não densa (acerto)", "em uso": float(((a >= 3) == (v >= 3)).mean()),
                        "candidato": float(((c >= 3) == (v >= 3)).mean()), "casos": len(d)}]
    elif modelo == "achado":
        d = b[(b["achado_confirmado"] != "") & (b["exame_confirmacao"] != "") & (b["achado_sombra"] != "")]
        if len(d):
            linhas.append({"medida": "achado (acerto)", "em uso": float((d["achado_modelo"] == d["achado_confirmado"]).mean()),
                           "candidato": float((d["achado_sombra"] == d["achado_confirmado"]).mean()), "casos": len(d)})
    elif modelo == "detector":
        reais = b[(b["lesao_real"] == "sim") & (b["sombra_detectou"] != "")]
        if len(reais):
            achou_uso = (reais["origem_lesao"] == "detectada pelo modelo") | \
                (reais.get("concordancia", "") == "o modelo também detectou")
            linhas.append({"medida": "lesões reais encontradas", "em uso": float(achou_uso.mean()),
                           "candidato": float((reais["sombra_detectou"] == "sim").mean()), "casos": len(reais)})
        revis = b[(b["origem_lesao"] == "detectada pelo modelo") & b["lesao_real"].isin(["sim", "não"])]
        if len(revis):
            imgs = revis.groupby(["rodada", "imagem"])
            falsas_uso = imgs.apply(lambda g: int((g["lesao_real"] == "não").sum()))
            falsas_cand = []
            for (rod, img), g in imgs:
                todas = b[(b["rodada"] == rod) & (b["imagem"] == img)]
                n_cand = _num(todas["caixas_sombra"]).max()
                reais_cand = int(((todas["lesao_real"] == "sim") & (todas["sombra_detectou"] == "sim")).sum())
                falsas_cand.append(max(0.0, float(n_cand) - reais_cand) if not np.isnan(n_cand) else np.nan)
            linhas.append({"medida": "caixas falsas por imagem", "em uso": float(falsas_uso.mean()),
                           "candidato": float(np.nanmean(falsas_cand)) if len(falsas_cand) else np.nan,
                           "casos": int(len(falsas_uso))})
    return pd.DataFrame(linhas)
