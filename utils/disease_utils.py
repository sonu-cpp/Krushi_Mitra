import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np

# ── ResNet9 (self-contained, no utils.model dependency) ──────────────────────
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1  = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2  = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# ── 38 disease classes (exact order from Harvestify training) ────────────────
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# ── Disease info dictionary ───────────────────────────────────────────────────
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'display': 'Apple — Apple Scab',
        'severity': 'Moderate',
        'cause': 'Fungal infection caused by Venturia inaequalis.',
        'symptoms': 'Olive-green to black lesions on leaves and fruit surface.',
        'prevention': 'Apply fungicides early in the season. Remove infected leaves. Use resistant apple varieties.',
    },
    'Apple___Black_rot': {
        'display': 'Apple — Black Rot',
        'severity': 'High',
        'cause': 'Fungal disease caused by Botryosphaeria obtusa.',
        'symptoms': 'Brown spots on leaves; dark, sunken lesions on fruit.',
        'prevention': 'Prune infected branches. Apply copper-based fungicide. Destroy mummified fruit.',
    },
    'Apple___Cedar_apple_rust': {
        'display': 'Apple — Cedar Apple Rust',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Gymnosporangium juniperi-virginianae.',
        'symptoms': 'Bright orange-yellow spots on upper leaf surfaces.',
        'prevention': 'Remove nearby cedar/juniper trees. Apply fungicide at pink bud stage.',
    },
    'Apple___healthy': {
        'display': 'Apple — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Leaf appears healthy with no visible infections.',
        'prevention': 'Continue regular monitoring and good agricultural practices.',
    },
    'Blueberry___healthy': {
        'display': 'Blueberry — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Maintain soil pH between 4.5–5.5. Regular irrigation.',
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'display': 'Cherry — Powdery Mildew',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Podosphaera clandestina.',
        'symptoms': 'White powdery coating on young leaves and shoots.',
        'prevention': 'Apply sulfur-based fungicide. Ensure good air circulation. Avoid overhead irrigation.',
    },
    'Cherry_(including_sour)___healthy': {
        'display': 'Cherry — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Leaf appears healthy.',
        'prevention': 'Maintain good pruning practices and adequate nutrition.',
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'display': 'Corn — Gray Leaf Spot',
        'severity': 'High',
        'cause': 'Fungal disease caused by Cercospora zeae-maydis.',
        'symptoms': 'Rectangular, grayish-brown lesions parallel to leaf veins.',
        'prevention': 'Use resistant hybrids. Rotate crops. Apply foliar fungicide.',
    },
    'Corn_(maize)___Common_rust_': {
        'display': 'Corn — Common Rust',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Puccinia sorghi.',
        'symptoms': 'Small, circular to elongate, golden-brown pustules on leaves.',
        'prevention': 'Plant resistant varieties. Apply fungicide if infection is severe.',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'display': 'Corn — Northern Leaf Blight',
        'severity': 'High',
        'cause': 'Fungal disease caused by Exserohilum turcicum.',
        'symptoms': 'Long, cigar-shaped, grayish-green to tan lesions on leaves.',
        'prevention': 'Use resistant hybrids. Rotate with non-host crops. Apply fungicide at early stages.',
    },
    'Corn_(maize)___healthy': {
        'display': 'Corn — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Maintain proper spacing and nutrition.',
    },
    'Grape___Black_rot': {
        'display': 'Grape — Black Rot',
        'severity': 'High',
        'cause': 'Fungal disease caused by Guignardia bidwellii.',
        'symptoms': 'Reddish-brown circular spots on leaves; shriveled black fruit.',
        'prevention': 'Remove mummified berries. Apply fungicide from early growth. Prune for air circulation.',
    },
    'Grape___Esca_(Black_Measles)': {
        'display': 'Grape — Esca (Black Measles)',
        'severity': 'High',
        'cause': 'Complex fungal disease involving several wood-rotting fungi.',
        'symptoms': 'Interveinal chlorosis and necrosis on leaves; dark berry spots.',
        'prevention': 'Avoid large pruning wounds. Apply wound sealants. Remove infected wood.',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'display': 'Grape — Leaf Blight',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Pseudocercospora vitis.',
        'symptoms': 'Irregular dark brown spots with yellow halos on leaves.',
        'prevention': 'Spray with copper-based fungicide. Improve air circulation.',
    },
    'Grape___healthy': {
        'display': 'Grape — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Vine appears healthy.',
        'prevention': 'Regular pruning and pest monitoring.',
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'display': 'Orange — Citrus Greening (HLB)',
        'severity': 'Critical',
        'cause': 'Bacterial disease caused by Candidatus Liberibacter spp., spread by psyllid insects.',
        'symptoms': 'Yellowing of shoots, misshapen and bitter fruits, leaf mottling.',
        'prevention': 'No cure exists. Remove infected trees. Control psyllid population with insecticides.',
    },
    'Peach___Bacterial_spot': {
        'display': 'Peach — Bacterial Spot',
        'severity': 'Moderate',
        'cause': 'Bacterial disease caused by Xanthomonas arboricola pv. pruni.',
        'symptoms': 'Water-soaked lesions on leaves that turn brown; fruit cracking.',
        'prevention': 'Apply copper-based bactericide. Use resistant varieties. Avoid wounding.',
    },
    'Peach___healthy': {
        'display': 'Peach — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Tree appears healthy.',
        'prevention': 'Maintain good sanitation and irrigation practices.',
    },
    'Pepper,_bell___Bacterial_spot': {
        'display': 'Bell Pepper — Bacterial Spot',
        'severity': 'Moderate',
        'cause': 'Caused by Xanthomonas euvesicatoria.',
        'symptoms': 'Small, water-soaked spots on leaves that turn brown with yellow halo.',
        'prevention': 'Use disease-free seeds. Apply copper bactericide. Avoid overhead irrigation.',
    },
    'Pepper,_bell___healthy': {
        'display': 'Bell Pepper — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Maintain good soil drainage and crop rotation.',
    },
    'Potato___Early_blight': {
        'display': 'Potato — Early Blight',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Alternaria solani.',
        'symptoms': 'Dark brown concentric ring spots on older leaves.',
        'prevention': 'Apply fungicide (mancozeb/chlorothalonil). Remove infected foliage. Use certified seeds.',
    },
    'Potato___Late_blight': {
        'display': 'Potato — Late Blight',
        'severity': 'Critical',
        'cause': 'Caused by Phytophthora infestans (same pathogen as Irish Famine).',
        'symptoms': 'Water-soaked dark lesions on leaves; white mold under leaves in humid conditions.',
        'prevention': 'Use resistant varieties. Apply systemic fungicide. Destroy infected plants immediately.',
    },
    'Potato___healthy': {
        'display': 'Potato — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Use certified disease-free seed tubers.',
    },
    'Raspberry___healthy': {
        'display': 'Raspberry — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Ensure good drainage and remove old canes after fruiting.',
    },
    'Soybean___healthy': {
        'display': 'Soybean — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Rotate with non-legume crops. Monitor for pests regularly.',
    },
    'Squash___Powdery_mildew': {
        'display': 'Squash — Powdery Mildew',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Podosphaera xanthii.',
        'symptoms': 'White powdery patches on leaf surfaces.',
        'prevention': 'Apply neem oil or sulfur fungicide. Improve air circulation.',
    },
    'Strawberry___Leaf_scorch': {
        'display': 'Strawberry — Leaf Scorch',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Diplocarpon earliana.',
        'symptoms': 'Small, dark purple spots on upper leaf surface; leaves turn brown.',
        'prevention': 'Apply fungicide. Remove infected leaves. Avoid overhead watering.',
    },
    'Strawberry___healthy': {
        'display': 'Strawberry — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Maintain good mulching and drip irrigation.',
    },
    'Tomato___Bacterial_spot': {
        'display': 'Tomato — Bacterial Spot',
        'severity': 'Moderate',
        'cause': 'Caused by Xanthomonas vesicatoria.',
        'symptoms': 'Small, dark, water-soaked spots on leaves and fruit.',
        'prevention': 'Use copper bactericide. Avoid working in wet fields. Use disease-free transplants.',
    },
    'Tomato___Early_blight': {
        'display': 'Tomato — Early Blight',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Alternaria solani.',
        'symptoms': 'Dark concentric ring spots (target-like) on older leaves.',
        'prevention': 'Rotate crops. Remove lower infected leaves. Apply mancozeb fungicide.',
    },
    'Tomato___Late_blight': {
        'display': 'Tomato — Late Blight',
        'severity': 'Critical',
        'cause': 'Caused by Phytophthora infestans.',
        'symptoms': 'Greasy, dark lesions on leaves; white mold visible in humid weather.',
        'prevention': 'Apply systemic fungicide immediately. Remove infected plants. Avoid overhead irrigation.',
    },
    'Tomato___Leaf_Mold': {
        'display': 'Tomato — Leaf Mold',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Passalora fulva.',
        'symptoms': 'Yellow patches on upper leaf; olive-brown mold on underside.',
        'prevention': 'Reduce humidity. Improve ventilation. Apply fungicide.',
    },
    'Tomato___Septoria_leaf_spot': {
        'display': 'Tomato — Septoria Leaf Spot',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Septoria lycopersici.',
        'symptoms': 'Circular spots with dark border and light center; small dark specks inside spots.',
        'prevention': 'Remove infected lower leaves. Apply chlorothalonil fungicide.',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'display': 'Tomato — Spider Mites',
        'severity': 'Moderate',
        'cause': 'Infestation by Tetranychus urticae (two-spotted spider mite).',
        'symptoms': 'Stippled, yellowing leaves; fine webbing on undersides.',
        'prevention': 'Apply miticide or neem oil. Maintain plant moisture. Introduce natural predators.',
    },
    'Tomato___Target_Spot': {
        'display': 'Tomato — Target Spot',
        'severity': 'Moderate',
        'cause': 'Fungal disease caused by Corynespora cassiicola.',
        'symptoms': 'Brown spots with concentric rings resembling a target.',
        'prevention': 'Apply fungicide (azoxystrobin). Remove infected leaves. Improve air circulation.',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'display': 'Tomato — Yellow Leaf Curl Virus',
        'severity': 'Critical',
        'cause': 'Viral disease spread by whiteflies (Bemisia tabaci).',
        'symptoms': 'Upward curling and yellowing of young leaves; stunted plant growth.',
        'prevention': 'Control whitefly with insecticides. Use virus-resistant varieties. Remove infected plants.',
    },
    'Tomato___Tomato_mosaic_virus': {
        'display': 'Tomato — Mosaic Virus',
        'severity': 'High',
        'cause': 'Caused by Tomato mosaic virus (ToMV), spread by contact.',
        'symptoms': 'Mosaic pattern of light and dark green on leaves; leaf distortion.',
        'prevention': 'Use virus-free seeds. Disinfect tools. Remove and destroy infected plants.',
    },
    'Tomato___healthy': {
        'display': 'Tomato — Healthy',
        'severity': 'None',
        'cause': 'No disease detected.',
        'symptoms': 'Plant appears healthy.',
        'prevention': 'Continue regular scouting and good agricultural practices.',
    },
}

SEVERITY_COLOR = {
    'None': '#16a34a',
    'Moderate': '#ca8a04',
    'High': '#dc2626',
    'Critical': '#7c3aed',
}

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_disease_model.pth')

_disease_model = None

def load_disease_model():
    global _disease_model
    if _disease_model is None:
        model = ResNet9(3, len(DISEASE_CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        _disease_model = model
    return _disease_model

# ── Transform (must match training exactly) ───────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Leaf validation (HSV green check) ────────────────────────────────────────
def is_leaf_image(image: Image.Image, threshold: float = 0.08) -> bool:
    """Returns True if the image contains enough green vegetation to be a leaf."""
    img_resized = image.resize((128, 128))
    img_array   = np.array(img_resized).astype(np.float32) / 255.0
    r, g, b     = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    # Green pixels: green channel dominates both red and blue
    green_mask  = (g > r + 0.05) & (g > b + 0.05) & (g > 0.2)
    green_ratio = green_mask.sum() / green_mask.size
    return green_ratio >= threshold

# ── Inference ─────────────────────────────────────────────────────────────────
class NotALeafError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass

def predict_disease(image_bytes):
    """
    Takes raw image bytes, returns (class_key, display_name, confidence_pct, info_dict).
    Raises NotALeafError if image doesn't look like a leaf.
    Raises ModelNotFoundError if model file is missing.
    """
    # ── Check model file exists ───────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please train the model first using train_disease_model.py"
        )

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise ValueError("Could not read the uploaded image. Please upload a valid JPG or PNG file.")

    # ── Leaf validation ───────────────────────────────────────────────────────
    if not is_leaf_image(image):
        raise NotALeafError(
            "The uploaded image does not appear to be a plant leaf. "
            "Please upload a clear photo of an affected leaf."
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    model = load_disease_model()
    img_t = _TRANSFORM(image)
    img_u = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        outputs       = model(img_u)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    class_key      = DISEASE_CLASSES[pred_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    info = DISEASE_INFO.get(class_key, {
        'display':    class_key.replace('___', ' — ').replace('_', ' '),
        'severity':   'Unknown',
        'cause':      'Information not available.',
        'symptoms':   'Refer to an agricultural expert.',
        'prevention': 'Consult your local agricultural extension officer.',
    })

    return class_key, info['display'], confidence_pct, info
