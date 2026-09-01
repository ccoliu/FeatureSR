"""
utils/prompt_templates.py
領域自適應 Prompt 工程與多模板集成 (Domain-Specific Multi-Prompt Ensembling)
"""
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
import clip

# ─────────────────────────────────────────────
# 1. 各資料集類別名稱的專業語意展開 (Domain Semantic Expansion)
# ─────────────────────────────────────────────
DOMAIN_CLASS_EXPANSIONS: Dict[str, Dict[str, str]] = {
    "isic": {
        "MEL": "melanoma, a malignant pigmented skin cancer lesion",
        "NV": "melanocytic nevus, a benign pigmented skin mole",
        "BCC": "basal cell carcinoma, a common type of skin cancer",
        "AKIEC": "actinic keratosis or intraepithelial carcinoma lesion",
        "BKL": "benign keratosis lesion, including seborrheic keratosis",
        "DF": "dermatofibroma, a benign fibrous nodule on the skin",
        "VASC": "vascular skin lesion, including angioma and pyogenic granuloma",
    },
    "chestx": {
        "Atelectasis": "atelectasis, partial or complete lung collapse",
        "Cardiomegaly": "cardiomegaly, enlarged cardiac heart silhouette",
        "Effusion": "pleural effusion, abnormal fluid accumulation in the chest pleural cavity",
        "Infiltration": "pulmonary infiltration, abnormal opacity in lung parenchyma tissue",
        "Mass": "pulmonary mass, abnormal chest lesion or tumor larger than 3 centimeters",
        "Nodule": "pulmonary nodule, discrete focal lung opacity or small lesion",
        "Pneumothorax": "pneumothorax, abnormal air collection in the pleural space",
    },
    "eurosat": {
        "AnnualCrop": "annual agricultural crop farmland and cultivated fields",
        "Forest": "forest woodland with dense trees and green canopy",
        "HerbaceousVegetation": "herbaceous natural green vegetation, grassland, and meadows",
        "Highway": "highway road infrastructure, interstate, and asphalt roadway",
        "Industrial": "industrial buildings, commercial warehouses, and factory facilities",
        "Pasture": "pasture grassland for grazing livestock and open grassy fields",
        "PermanentCrop": "permanent crop orchards, vineyards, and fruit tree plantations",
        "Residential": "residential houses, neighborhood buildings, and suburban living area",
        "River": "river waterway, streaming water canal, and freshwater channel",
        "SeaLake": "sea, ocean, lake, or large body of surface water",
    },
}

# ─────────────────────────────────────────────
# 2. 各領域專屬多模板 (Domain Multi-Templates)
# ─────────────────────────────────────────────
DOMAIN_TEMPLATES: Dict[str, List[str]] = {
    "isic": [
        "a dermatoscopic photograph of {c}.",
        "a dermatoscopy image showing {c}.",
        "a clinical close-up photo of {c} on human skin.",
        "a medical skin lesion photograph displaying {c}.",
        "a high-resolution dermoscopic image showing {c}.",
        "a dermatological examination photo of {c}.",
        "a microscopic pathology image of skin {c}.",
        "a photograph of human skin affected by {c}.",
    ],
    "chestx": [
        "a chest X-ray radiograph showing {c}.",
        "a frontal chest radiograph with evidence of {c}.",
        "a medical radiograph displaying pulmonary {c}.",
        "a chest radiography image showing {c} pathology.",
        "a clinical X-ray radiograph indicating {c}.",
        "a chest radiograph with radiographic findings of {c}.",
        "a radiologist examination X-ray showing {c}.",
    ],
    "eurosat": [
        "a satellite photo of {c}.",
        "an aerial satellite view of {c}.",
        "a remote sensing satellite image showing {c}.",
        "an overhead satellite photograph displaying {c}.",
        "a satellite imagery of earth surface with {c}.",
        "an aerial top-down view of {c}.",
        "a multispectral satellite earth observation of {c}.",
    ],
    "crop_disease": [
        "a photo of a plant leaf with {c}.",
        "a close-up photograph of a diseased crop leaf showing {c}.",
        "an agricultural photograph of a leaf affected by {c}.",
        "a plant pathology image showing {c} on foliage.",
        "a photo of a crop plant leaf exhibiting symptoms of {c}.",
        "a macro photo of plant leaf infected with {c}.",
        "a botanical photo of {c} on leaf surface.",
    ],
    "generic": [
        "a photo of a {c}.",
        "a picture of a {c}.",
        "a close-up photo of a {c}.",
        "an image showing a {c}.",
    ]
}


def clean_class_name(name: str, dataset_name: Optional[str] = None) -> str:
    """清理並展開類別名稱"""
    ds = (dataset_name or "").lower()
    
    # 1. 優先查表展開專屬名詞
    if ds in DOMAIN_CLASS_EXPANSIONS and name in DOMAIN_CLASS_EXPANSIONS[ds]:
        return DOMAIN_CLASS_EXPANSIONS[ds][name]
    
    # 2. 針對 CropDiseases 特殊命名清洗 (Apple___Apple_scab -> apple leaf with apple scab)
    if ds == "crop_disease" or "___" in name:
        parts = name.split("___")
        if len(parts) == 2:
            plant = parts[0].replace("_", " ").strip()
            disease = parts[1].replace("_", " ").strip()
            if disease.lower() == "healthy":
                return f"healthy fresh {plant} leaf"
            return f"{plant} leaf with {disease}"
    
    # 3. 通用字串清洗
    return name.replace("___", " ").replace("__", " ").replace("_", " ").strip()


def get_domain_text_embeddings(
    clip_model,
    class_names: List[str],
    dataset_name: str,
    device: str = "cuda",
    use_ensemble: bool = True,
) -> torch.Tensor:
    """
    生成領域自適應且集成多模板的文字特徵向量 (Ensemble Text Embeddings)。

    Args:
        clip_model: CLIP 模型或 CLIPWrapper (具有 encode_text 方法)
        class_names: 原始類別名稱清單 [C]
        dataset_name: 資料集名稱 (isic, chestx, eurosat, crop_disease)
        device: 運算設備
        use_ensemble: 是否使用多模板集成 (True = 平均多個 Prompt 嵌入, False = 單一 Prompt)

    Returns:
        text_feat: [C, d] L2-normalized 領域增強文字特徵
    """
    ds = dataset_name.lower()
    templates = DOMAIN_TEMPLATES.get(ds, DOMAIN_TEMPLATES["generic"])
    
    if not use_ensemble:
        templates = [templates[0]]
        
    class_embeddings = []
    
    for name in class_names:
        c_clean = clean_class_name(name, ds)
        prompts = [tpl.format(c=c_clean) for tpl in templates]
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        
        with torch.no_grad():
            # 兼容原生 clip.model 與自訂 wrapper
            if hasattr(clip_model, "model"):
                emb = clip_model.model.encode_text(tokens)
            else:
                emb = clip_model.encode_text(tokens)
                
            # 各模板特徵先做 L2 正規化
            emb = F.normalize(emb.float(), dim=-1)
            # 多模板取平均 (Ensemble Pooling)
            mean_emb = emb.mean(dim=0, keepdim=True)
            # 最終再次 L2 正規化
            mean_emb = F.normalize(mean_emb, dim=-1)
            class_embeddings.append(mean_emb)
            
    text_feat = torch.cat(class_embeddings, dim=0)  # [C, d]
    return text_feat
