"""
Ponte entre o MARIA e o GMIC (NYU; Shen et al., "An interpretable classifier for high-resolution breast cancer
screening images utilizing weakly supervised localization", Medical Image Analysis 68, 2021).

O GMIC olha a mamografia INTEIRA e devolve a P(maligno) da imagem e um mapa de saliência ("onde o modelo olhou").
No MARIA ele entra só na PRIORIDADE DA MAMA (média com a patologia do MARIA, ver calibracao.json -> "hibrido_gmic");
o detalhe por lesão continua com o detector e a patologia do MARIA.

LICENÇA: o código e os pesos do GMIC são GNU AGPLv3 e NÃO fazem parte do MARIA. Eles ficam em
modelo/modulo_gmic/, baixados à parte por config/baixar_gmic.py (versão fixa, pesos conferidos por SHA-256).
Este arquivo (a ponte) é do MARIA (MIT). Sem o módulo, o MARIA funciona como antes (prioridade só do MARIA).
"""
import sys
import types
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent / "modulo_gmic"
COMMIT = "594029efe6184bcca3a151fa1dd46c5fd620cb25"          # nyukat/GMIC, versão testada
URL_ZIP = f"https://github.com/nyukat/GMIC/archive/{COMMIT}.zip"
HASHES = {                                                   # SHA-256 dos 5 pesos publicados (conferidos ao baixar)
    "sample_model_1.p": "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991",
    "sample_model_2.p": "b09da8d5c128f42f2a2589bfec5ffa28b16cb3d00fbc5d45913cbb402d05754e",
    "sample_model_3.p": "07800b83f06eea69c6afe05ecd83523026c391762fb6593538623acd9e715bbd",
    "sample_model_4.p": "775b3e29301fdf7d33f16f4ee5f5a5338a23d858572c66cb0097f678cd2cb4c1",
    "sample_model_5.p": "8209bef18775419e6d5575ae73a761458382aa342809fc3ff83b73e6c1fcf585",
}
VERSAO = "gmic-" + COMMIT[:7]
TAMANHO = (2944, 1920)                                       # janela de entrada do GMIC (altura, largura)
PARAMS = {"device_type": "cpu", "gpu_number": 0, "max_crop_noise": (100, 100), "max_crop_size_noise": 100,
          "cam_size": (46, 30), "K": 6, "crop_shape": (256, 256), "post_processing_dim": 256, "num_classes": 2,
          "use_v1_global": False}


def disponivel():
    return (PASTA / "src" / "modeling" / "gmic.py").is_file() and \
        all((PASTA / "models" / f"sample_model_{k}.p").is_file() for k in range(1, 6))


def conferir():
    """-> lista de problemas (vazia = código presente e os 5 pesos com o SHA-256 esperado)."""
    import hashlib
    ruins = []
    for nome, h in HASHES.items():
        arq = PASTA / "models" / nome
        if not arq.is_file():
            ruins.append(f"{nome}: falta")
        elif hashlib.sha256(arq.read_bytes()).hexdigest() != h:
            ruins.append(f"{nome}: SHA-256 diferente")
    if not (PASTA / "src" / "modeling" / "gmic.py").is_file():
        ruins.append("código (src/) faltando")
    return ruins


def _modulos():
    """Importa as partes do GMIC usadas aqui (só inferência). h5py/imageio só servem para ler/gravar arquivos no
    repositório original; se faltarem, entram módulos vazios no lugar."""
    for nome in ("h5py", "imageio"):
        try:
            __import__(nome)
        except ImportError:
            sys.modules[nome] = types.ModuleType(nome)
    import scipy.ndimage
    if not hasattr(scipy.ndimage, "morphology"):        # scipy novo: o submódulo antigo usado pelo GMIC sumiu
        scipy.ndimage.morphology = scipy.ndimage
    if str(PASTA) not in sys.path:
        sys.path.insert(0, str(PASTA))
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="src")
    from src.constants import PERCENT_T_DICT
    from src.cropping import crop_mammogram
    from src.data_loading import augmentations, loading
    from src.modeling import gmic
    from src.optimal_centers.get_optimal_centers import extract_center
    return types.SimpleNamespace(PERCENT_T_DICT=PERCENT_T_DICT, crop=crop_mammogram, aug=augmentations,
                                 loading=loading, gmic=gmic, extract_center=extract_center)


def parede_a_direita(arr):
    """A parede torácica fica do lado com mais mama (o tecido encosta na parede)."""
    mn, mx = float(arr.min()), float(arr.max())
    tecido = arr > mn + 0.10 * (mx - mn)
    meio = arr.shape[1] // 2
    return bool(tecido[:, meio:].sum() > tecido[:, :meio].sum())


def _fundo_zero(arr):
    """O corte do GMIC acha a mama pelos pixels > 0. Imagens sem fundo exatamente preto (filme digitalizado, JPG)
    têm o fundo zerado abaixo de 5% da faixa; as que já têm fundo preto ficam como estão."""
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    arr = arr - mn
    if (arr == 0).mean() < 0.10:
        arr[arr < 0.05 * (mx - mn)] = 0
    return arr


class GMIC:
    def __init__(self, device):
        import torch
        self.torch, self.device, self.m = torch, device, _modulos()
        self.redes = []
        for k in range(1, 6):
            p = dict(PARAMS, percent_t=self.m.PERCENT_T_DICT[str(k)],
                     device_type="gpu" if device.type == "cuda" else "cpu")
            rede = self.m.gmic.GMIC(p)
            estado = torch.load(PASTA / "models" / f"sample_model_{k}.p", map_location=device, weights_only=True)
            rede.load_state_dict(estado, strict=False)
            self.redes.append(rede.to(device).eval())

    def preparar(self, arr, vista="CC"):
        """Mamografia completa (qualquer lado) -> entrada do GMIC + o que é preciso para devolver o mapa à imagem.
        Tudo é posto na orientação de uma mama ESQUERDA (parede à esquerda), a que o GMIC usa sem espelhar."""
        m = self.m
        espelhar = parede_a_direita(arr)
        img = _fundo_zero(np.fliplr(arr) if espelhar else arr)
        v = "MLO" if str(vista).upper() == "MLO" else "CC"
        t, b, l, r = (info := m.crop.crop_img_from_largest_connected(img, "right", True, 100, 50, 1 / 3))[0]
        corte = np.ascontiguousarray(img[t:b, l:r])
        dado = {"full_view": f"L-{v}", "view": v, "horizontal_flip": "NO",
                "rightmost_points": info[1], "bottommost_points": info[2]}
        centro = m.extract_center(dado, corte)
        x = m.loading.process_image(corte.copy(), f"L-{v}", centro)
        # mesma janela que process_image usou (sem ruído): para recolocar o mapa de saliência na imagem
        _, bordas = m.aug.sample_crop_best_center(corte[..., None], TAMANHO, np.random.RandomState(0), (0, 0), 0,
                                                  centro, f"L-{v}")
        pad_topo = int((TAMANHO[0] - corte.shape[0] + 0) / 2) if v == "CC" and corte.shape[0] < TAMANHO[0] else 0
        return x, {"espelhar": espelhar, "corte": (t, b, l, r), "janela": tuple(int(q) for q in bordas),
                   "pad_topo": pad_topo, "forma": arr.shape}

    def prever(self, arr, vista="CC", com_mapa=True):
        """-> (P(maligno) média das 5 redes, desvio entre as redes, mapa de saliência na forma de arr ou None)."""
        torch = self.torch
        x, info = self.preparar(arr, vista)
        tx = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(self.device)
        ps, mapas = [], []
        with torch.no_grad():
            for rede in self.redes:
                ps.append(float(rede(tx)[0, 1]))
                if com_mapa:
                    mapas.append(rede.saliency_map[0, 1].float())
        mapa = self._na_imagem(torch.stack(mapas).mean(0), info) if com_mapa else None
        return float(np.mean(ps)), float(np.std(ps)), mapa

    def _na_imagem(self, sal, info):
        """Mapa 46x30 da janela -> mapa do tamanho da mamografia original (0 fora da janela)."""
        import torch.nn.functional as F
        cheio = F.interpolate(sal[None, None], size=TAMANHO, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        t, b, l, r = info["corte"]
        wt, wb, wl, wr = info["janela"]
        H, W = info["forma"]
        mapa = np.zeros((H, W), np.float32)
        y0 = t + wt - info["pad_topo"]                 # topo da janela na imagem (já espelhada)
        x0 = l + wl
        ys0, xs0 = max(0, y0), max(0, x0)
        ys1, xs1 = min(H, y0 + TAMANHO[0], t + (b - t)), min(W, x0 + TAMANHO[1], r)
        if ys1 > ys0 and xs1 > xs0:
            mapa[ys0:ys1, xs0:xs1] = cheio[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        return np.fliplr(mapa) if info["espelhar"] else mapa
