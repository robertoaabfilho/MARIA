"""
Pseudonimização e anonimização (LGPD) para dados que SAEM do atendimento: banco de anotações, exportação .zip
e retreino. (Os resultados do dia a dia continuam identificados, porque o radiologista precisa saber de quem é
o exame; ficam protegidos por login e auditoria.)

- pseudonimizar(texto): HMAC-SHA256 com uma chave local (config/chave_pseudonimizacao.key, fora do git).
  O mesmo identificador vira sempre o mesmo pseudônimo (dá para juntar exames da mesma mama), mas não dá para
  voltar ao original sem a chave.
- anonimizar_dicom(origem, destino): remove nome, ID, data de nascimento, endereço, instituição, médicos,
  números de acesso, datas e tags privadas; troca os UIDs por novos (determinísticos pela chave) e marca
  PatientIdentityRemoved. Mantém idade, sexo, fabricante, lateralidade e vista (usados nas análises).
  LIMITAÇÃO: textos gravados DENTRO da imagem (pixels) não são removidos.
- copiar_anonimizado(origem, destino): DICOM -> anonimizar_dicom; PNG/JPG -> regrava sem metadados.
"""
import hashlib
import hmac
import os
import secrets
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ_CHAVE = RAIZ / "config" / "chave_pseudonimizacao.key"
TAGS_REMOVER = ["PatientName", "PatientBirthDate", "PatientBirthTime", "PatientAddress", "PatientTelephoneNumbers",
                "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence", "PatientMotherBirthName",
                "MilitaryRank", "EthnicGroup", "PatientComments", "InstitutionName", "InstitutionAddress",
                "InstitutionalDepartmentName", "ReferringPhysicianName", "ReferringPhysicianAddress",
                "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "OperatorsName", "PhysiciansOfRecord",
                "RequestingPhysician", "AccessionNumber", "StudyID", "StationName", "DeviceSerialNumber",
                "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime", "SeriesTime",
                "AcquisitionTime", "ContentTime", "AcquisitionDateTime", "InstanceCreationDate",
                "InstanceCreationTime", "RequestAttributesSequence", "ReferencedPatientSequence"]
TAGS_UID = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"]


def chave():
    if not ARQ_CHAVE.is_file():
        ARQ_CHAVE.parent.mkdir(parents=True, exist_ok=True)
        ARQ_CHAVE.write_text(secrets.token_hex(32), encoding="utf-8")
        try:
            os.chmod(ARQ_CHAVE, 0o600)
        except OSError:
            pass
    return bytes.fromhex(ARQ_CHAVE.read_text(encoding="utf-8").strip())


def pseudonimizar(texto, prefixo="ID"):
    if texto is None or str(texto).strip() == "":
        return ""
    h = hmac.new(chave(), str(texto).encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"{prefixo}-{h}"


def anonimizar_dicom(origem, destino):
    import pydicom
    from pydicom.uid import generate_uid
    ds = pydicom.dcmread(str(origem))
    for tag in TAGS_REMOVER:
        if tag in ds:
            delattr(ds, tag)
    if "PatientID" in ds:
        ds.PatientID = pseudonimizar(ds.PatientID, "PAC")
    for tag in TAGS_UID:
        if tag in ds:
            setattr(ds, tag, generate_uid(entropy_srcs=[chave().hex(), str(getattr(ds, tag))]))
    if hasattr(ds, "file_meta") and "SOPInstanceUID" in ds:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "MARIA: ids, datas e tags privadas removidos; UIDs novos"   # VR LO: até 64
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(destino))


def copiar_anonimizado(origem, destino):
    origem, destino = Path(origem), Path(destino)
    if origem.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = Image.open(origem)
        limpa = img.copy()
        limpa.info = {}                               # descarta EXIF, textos e comentários embutidos
        destino.parent.mkdir(parents=True, exist_ok=True)
        limpa.save(destino)
    else:
        anonimizar_dicom(origem, destino)
