import streamlit as st
import pickle
import pandas as pd
from googletrans import Translator
import shap
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Language code mapping
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Odia": "or",
    "Tamil": "ta",
    "Telugu": "te"
}

# --------------------
# Translator function
# --------------------
translator = Translator()

def translate_text(text, lang):
    code = lang_map.get(lang, "en")
    if code == "en":
        return text
    try:
        return translator.translate(text, dest=code).text
    except:
        return text

# --------------------
# Load models
# --------------------
crop_model = pickle.load(open("models/crop_model.pkl", "rb"))
disease_model = tf.keras.models.load_model("models/disease_model.h5")

# Disease classes (based on dataset folders)
disease_classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Disease cures
disease_cures = {
    'Pepper__bell___Bacterial_spot': "Remove infected leaves and fruits. Apply copper-based fungicides. Avoid overhead watering. Plant resistant varieties.",
    'Pepper__bell___healthy': "No treatment needed. Keep monitoring for early signs.",
    'Potato___Early_blight': "Apply fungicides like chlorothalonil. Remove affected leaves. Ensure good air circulation. Rotate crops.",
    'Potato___healthy': "No treatment needed. Maintain soil health.",
    'Potato___Late_blight': "Use fungicides immediately. Destroy infected plants. Avoid wet conditions. Use resistant varieties.",
    'Tomato__Target_Spot': "Apply fungicides. Remove infected leaves. Improve air circulation. Avoid wetting leaves.",
    'Tomato__Tomato_mosaic_virus': "Remove infected plants. Control aphids (virus carriers). Use virus-free seeds. Disinfect tools.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Control whiteflies with insecticides. Use reflective mulches. Plant resistant varieties.",
    'Tomato_Bacterial_spot': "Apply copper fungicides. Avoid overhead irrigation. Remove infected plants. Use disease-free seeds.",
    'Tomato_Early_blight': "Use fungicides. Remove affected leaves. Mulch to prevent soil splash. Rotate crops.",
    'Tomato_healthy': "No treatment needed. Regular monitoring advised.",
    'Tomato_Late_blight': "Apply fungicides promptly. Remove infected parts. Ensure drainage. Use resistant varieties.",
    'Tomato_Leaf_Mold': "Improve ventilation. Apply fungicides. Avoid wetting leaves. Remove infected leaves.",
    'Tomato_Septoria_leaf_spot': "Use fungicides. Remove lower leaves. Mulch around plants. Rotate crops.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Spray with insecticidal soap or neem oil. Increase humidity. Introduce predatory mites."
}

# --------------------
# SHAP Explainer for Crop
# --------------------
crop_explainer = shap.TreeExplainer(crop_model)

# --------------------
# Grad-CAM function for Disease
# --------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            return layer.name
    return None

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="🌾 AI Farmer Advisor", page_icon="🌾", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.8em;
        color: #228B22;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #228B22;
    }
    .warning-box {
        background-color: #fffacd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #32cd32;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌾 AI Based Advisor System for farmers </div>', unsafe_allow_html=True)
st.markdown("### 🤖 Your Smart Farming Companion!")
st.markdown("This AI-powered app helps farmers like you make better decisions. Choose your language and explore features below.")

lang = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Odia", "Tamil", "Telugu"])

# Sidebar for page selection
page = st.sidebar.selectbox(translate_text("🌟 Choose Feature", lang), ["🏠 Home", "🌾 Crop Recommendation", "🦠 Disease Detection", "📚 Farming Tips", "ℹ️ About"])

# General farming tips
farming_tips = {
    "English": [
        "🌱 Always test your soil before planting to know nutrient levels.",
        "💧 Water your crops early in the morning to reduce evaporation.",
        "🐛 Check plants regularly for pests and diseases.",
        "🌞 Plant crops according to the season for better yield.",
        "🧪 Use organic fertilizers to keep soil healthy."
    ],
    "Hindi": [
        "🌱 फसल लगाने से पहले हमेशा अपनी मिट्टी का परीक्षण करें।",
        "💧 सुबह जल्दी पानी दें ताकि वाष्पीकरण कम हो।",
        "🐛 पौधों को नियमित रूप से कीट और बीमारियों के लिए जांचें।",
        "🌞 बेहतर उपज के लिए मौसम के अनुसार फसल लगाएं।",
        "🧪 मिट्टी को स्वस्थ रखने के लिए जैविक उर्वरक का उपयोग करें।"
    ],
    "Odia": [
        "🌱 ଚାଷ କରିବା ପୂର୍ବରୁ ସର୍ବଦା ଆପଣଙ୍କ ମାଟିର ପରୀକ୍ଷଣ କରନ୍ତୁ।",
        "💧 ବାଷ୍ପୀକରଣ କମ କରିବା ପାଇଁ ସକାଳରେ ପାଣି ଦିଅନ୍ତୁ।",
        "🐛 କୀଟ ଏବଂ ରୋଗ ପାଇଁ ନିୟମିତ ଭାବରେ ପୌଧକୁ ଯାଞ୍ଚ କରନ୍ତୁ।",
        "🌞 ଭଲ ଉତ୍ପାଦନ ପାଇଁ ଋତୁ ଅନୁସାରେ ଫସଲ ଲଗାନ୍ତୁ।",
        "🧪 ମାଟିକୁ ସ୍ୱାସ୍ଥ୍ୟକର ରଖିବା ପାଇଁ ଜୈବିକ ଖାଦ୍ୟ ବ୍ୟବହାର କରନ୍ତୁ।"
    ],
    "Tamil": [
        "🌱 விவசாயம் செய்வதற்கு முன் எப்போதும் உங்கள் மண்ணை சோதிக்கவும்.",
        "💧 ஆவியாதலை குறைக்க காலையில் தண்ணீர் ஊற்றவும்.",
        "🐛 பூச்சிகள் மற்றும் நோய்களுக்கு தாவரங்களை தொடர்ந்து சரிபார்க்கவும்.",
        "🌞 சிறந்த விளைச்சலுக்கு பருவத்திற்கு ஏற்ப பயிர் செய்யவும்.",
        "🧪 மண்ணை ஆரோக்கியமாக வைத்திருக்க ஜீவாமற்ற உரங்களைப் பயன்படுத்தவும்."
    ],
    "Telugu": [
        "🌱 వ్యవసాయం చేయడానికి ముందు ఎల్లప్పుడూ మీ మట్టిని పరీక్షించండి.",
        "💧 ఆవిరిపోవడం తగ్గించడానికి తెల్లవారుజామున నీరు పెట్టండి.",
        "🐛 పురుగులు మరియు వ్యాధుల కోసం మొక్కలను నియమితంగా తనిఖీ చేయండి.",
        "🌞 మంచి పంట కోసం కాలానుగుణంగా పంటలు వేయండి.",
        "🧪 మట్టిని ఆరోగ్యంగా ఉంచడానికి సేంద్రియ ఎరువులను ఉపయోగించండి."
    ]
}

if page == "🏠 Home":
    st.markdown(f'<div class="sub-header">{translate_text("🏠 Home", lang)}</div>', unsafe_allow_html=True)
    st.markdown(translate_text("""
    Welcome to the AI Farmer Advisor! This app uses advanced AI to help you:
    - **Recommend the best crops** based on your soil and weather conditions.
    - **Detect plant diseases** from photos of your leaves.
    - **Get farming tips** in your preferred language.
    
    ### How to Use:
    1. Select your language from the dropdown above.
    2. Choose a feature from the sidebar.
    3. Follow the instructions on each page.
    
    ### Features:
    - 🌾 **Crop Recommendation**: Input soil nutrients and weather data to get crop suggestions with explanations.
    - 🦠 **Disease Detection**: Upload a clear photo of a plant leaf to identify diseases and get treatment advice.
    - 📚 **Farming Tips**: Read helpful tips in multiple languages.
    - ℹ️ **About**: Learn more about the app and its creators.
    
    Start by selecting a feature from the sidebar!
    """, lang), unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x400?text=Farming+with+AI", caption=translate_text("AI helping farmers grow better!", lang))

elif page == "🌾 Crop Recommendation":
    st.markdown(f'<div class="sub-header">{translate_text("🌾 Crop Recommendation", lang)}</div>', unsafe_allow_html=True)
    st.markdown(translate_text("""
    Get personalized crop recommendations based on your soil nutrients and weather conditions. 
    The AI analyzes factors like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall to suggest the best crop for your farm.
    
    ### Instructions:
    1. Adjust the sliders below with your soil test results and local weather data.
    2. Click "Get Recommendation" to see the suggested crop.
    3. View the explanation to understand why that crop is recommended.
    
    **Tip:** If you don't have exact values, use average estimates for your region.
    """, lang), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(translate_text("### 🌱 Soil Nutrients (kg/ha)", lang))
        st.markdown(translate_text("These are key elements for plant growth. Get tested at a local lab.", lang))
        N = st.slider(
            translate_text("Nitrogen (N) - For leaf growth", lang),
            min_value=0,
            max_value=140,
            value=50,
            help=translate_text("Nitrogen helps in green leafy growth. Typical range: 0-140 kg/ha", lang),
            key="N"
        )

        P = st.slider(
            translate_text("Phosphorus (P) - For root development", lang),
            min_value=0,
            max_value=145,
            value=50,
            help=translate_text("Phosphorus aids in root and flower development. Typical: 0-145 kg/ha", lang),
            key="P"
        )

        K = st.slider(
            translate_text("Potassium (K) - For disease resistance", lang),
            min_value=0,
            max_value=205,
            value=50,
            help=translate_text("Potassium improves disease resistance and fruit quality. Typical: 0-205 kg/ha", lang),
            key="K"
        )

        st.markdown(translate_text("### 🌡️ Weather Conditions", lang))
        temp = st.slider(
            translate_text("Temperature (°C) - Average daily", lang),
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            help=translate_text("Ideal temperature for most crops: 15-35°C. Check local weather apps.", lang),
            key="temp"
        )

        hum = st.slider(
            translate_text("Humidity (%) - Average", lang),
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            help=translate_text("Humidity affects plant health. Typical: 30-90%. High humidity can cause diseases.", lang),
            key="hum"
        )

        st.markdown("### 🏞️ Soil & Rain")
        ph = st.slider(
            max_value=205,
            value=50,
            help=translate_text("Slide to set potassium level. Typical: 0-205", lang),
            key="K"
        )

        st.markdown("### 🌡️ Weather Conditions")
        temp = st.slider(
            translate_text("Temperature (°C)", lang),
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            help=translate_text("Slide to set average temperature. Ideal: 15-35°C", lang),
            key="temp"
        )

        hum = st.slider(
            translate_text("Humidity (%)", lang),
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            help=translate_text("Slide to set humidity. Typical: 30-90%", lang),
            key="hum"
        )

        st.markdown(translate_text("### 🏞️ Soil & Rain", lang))
        ph = st.slider(
            translate_text("Soil pH", lang),
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            help=translate_text("Slide to set soil pH. Ideal: 5.5-7.5", lang),
            key="ph"
        )

        rain = st.slider(
            translate_text("Rainfall (mm)", lang),
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            help=translate_text("Slide to set annual rainfall in mm.", lang),
            key="rain"
        )

    with col2:
        st.info(translate_text("""
📌 **How to Enter Values**
- Nitrogen, Phosphorus, Potassium: Soil nutrient values (kg/ha)
- Temperature: Average temperature (°C)
- Humidity: Relative humidity (%)
- pH: Soil acidity/alkalinity
- Rainfall: Annual rainfall (mm)
""", lang))

    if st.button(translate_text("🌱 Predict My Best Crop!", lang), use_container_width=True):
        try:
            with st.spinner(translate_text("Analyzing your soil and weather data...", lang)):
                input_df = pd.DataFrame(
                    [[N, P, K, temp, hum, ph, rain]],
                    columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
                )

                crop = crop_model.predict(input_df)[0]
                st.success(translate_text(f"🎉 Recommended Crop: {crop}", lang))
                st.balloons()

                # Show crop image (placeholder)
                st.image(f"https://via.placeholder.com/300x200?text={crop}", caption=translate_text(f"Image of {crop}", lang))

                # Farming tip for the crop
                tip = translate_text(f"For {crop}, ensure proper irrigation and pest control.", lang)
                st.info(f"💡 {tip}")

                # SHAP Explanation
                try:
                    st.subheader(translate_text("🔍 Why this crop? (AI Explanation)", lang))
                    shap_values = crop_explainer.shap_values(input_df)
                    feature_names = [translate_text("Nitrogen", lang), translate_text("Phosphorus", lang), translate_text("Potassium", lang), translate_text("Temperature", lang), translate_text("Humidity", lang), translate_text("pH", lang), translate_text("Rainfall", lang)]

                    predicted_class_index = np.argmax(crop_model.predict_proba(input_df), axis=1)[0]
                    shap_vals = shap_values[predicted_class_index][0]

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(feature_names, shap_vals, color='lightgreen')
                    ax.set_xlabel(translate_text("Impact on Decision", lang))
                    ax.set_title(translate_text("Feature Importance", lang))
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(translate_text("Explanation chart could not be generated.", lang))
        except Exception as e:
            st.error(translate_text("Sorry, there was an error predicting the crop. Please check your inputs.", lang))

elif page == "🦠 Disease Detection":
    st.markdown(f'<div class="sub-header">{translate_text("🦠 Plant Disease Detection", lang)}</div>', unsafe_allow_html=True)
    st.markdown(translate_text("""
    Upload a photo of your plant leaf to detect diseases. The AI analyzes the image and provides diagnosis with treatment recommendations.
    
    ### Photo Tips for Best Results:
    - Take a clear, well-lit photo of the leaf.
    - Focus on the affected area if possible.
    - Avoid blurry or dark images.
    - Supported formats: JPG, PNG, JPEG.
    
    ### Instructions:
    1. Click "Browse files" to upload your photo.
    2. Click "Check for Disease" to analyze.
    3. View the results, heatmap, and cure advice.
    
    **Note:** This is for informational purposes. Consult a local expert for confirmation.
    """, lang), unsafe_allow_html=True)

    uploaded_file = st.file_uploader(translate_text("📷 Upload a clear photo of your plant leaf", lang), type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=translate_text("Your uploaded leaf photo", lang), use_column_width=True)

        if st.button(translate_text("🔍 Check for Disease", lang)):
            try:
                with st.spinner(translate_text("Analyzing the image...", lang)):
                    # Preprocess
                    img_array = preprocess_image(image)

                    # Predict
                    predictions = disease_model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    disease = disease_classes[predicted_class]
                    confidence = predictions[0][predicted_class] * 100

                    if confidence > 70:
                        st.success(translate_text(f"✅ Detected Disease: {disease} (Confidence: {confidence:.1f}%)", lang))
                        st.balloons()
                    else:
                        st.warning(translate_text(f"🤔 Possible Disease: {disease} (Low confidence: {confidence:.1f}%). Please try a clearer image.", lang))

                    # Show disease image (placeholder)
                    st.image(f"https://via.placeholder.com/300x200?text={disease.replace('___', ' ').replace('_', ' ')}", caption=translate_text(f"What {disease} looks like", lang))

                    # Cure/ Treatment
                    cure = disease_cures.get(disease, "Consult a local agricultural expert for specific treatment.")
                    st.info(translate_text(f"🩺 Recommended Cure/Treatment: {cure}", lang))

                    # Grad-CAM Explanation
                    try:
                        st.subheader(translate_text("🔍 AI Focus Areas (Heatmap)", lang))
                        last_conv = get_last_conv_layer(disease_model)
                        if last_conv:
                            heatmap = make_gradcam_heatmap(img_array, disease_model, last_conv)

                            if heatmap is not None:
                                # Superimpose heatmap on image
                                img = np.array(image.resize((224, 224)))
                                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                                heatmap = np.uint8(255 * heatmap)
                                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                                superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                                st.image(superimposed_img, caption=translate_text("Red areas show what AI focused on", lang), use_column_width=True)
                            else:
                                st.info(translate_text("Heatmap not available for this model.", lang))
                        else:
                            st.info(translate_text("No convolutional layers found for heatmap.", lang))
                    except Exception as e:
                        st.warning(translate_text("Heatmap could not be generated, but prediction is ready.", lang))
            except Exception as e:
                st.error(translate_text("Sorry, there was an error analyzing the image. Please try again with a different photo.", lang))

elif page == "📚 Farming Tips":
    st.markdown(f'<div class="sub-header">{translate_text("📚 Helpful Farming Tips", lang)}</div>', unsafe_allow_html=True)
    st.markdown(translate_text("""
    Here are some practical tips to improve your farming practices. These are translated into your selected language for easy understanding.
    
    ### Why These Tips?
    - Based on best agricultural practices.
    - Help in sustainable farming.
    - Easy to implement with low cost.
    
    Read and apply them to boost your yield!
    """, lang), unsafe_allow_html=True)

    st.markdown(translate_text("### 🌟 Essential Tips:", lang))
    for tip in farming_tips.get(lang, farming_tips["English"]):
        st.markdown(f"- {tip}")

    with st.expander(translate_text("More Advanced Tips", lang)):
        st.markdown(translate_text("""
        ### 🌟 More Help
        - Visit your local agriculture office for free advice.
        - Use weather apps to plan your work.
        - Join farmer groups to learn from others.
        - Test your soil regularly for best results.
        - Rotate crops to prevent soil depletion.
        - Use organic pesticides to protect beneficial insects.
        """, lang))

elif page == "ℹ️ About":
    st.markdown(f'<div class="sub-header">{translate_text("ℹ️ About This App", lang)}</div>', unsafe_allow_html=True)
    st.markdown(translate_text("""
    ### What is AI Farmer Advisor?
    This app is built using cutting-edge AI technologies to assist farmers in decision-making. It combines Machine Learning, Deep Learning, and Natural Language Processing for a comprehensive farming tool.
    
    ### Technologies Used:
    - **Streamlit**: For the user-friendly web interface.
    - **Scikit-learn**: For crop recommendation model.
    - **TensorFlow/Keras**: For disease detection CNN.
    - **SHAP & Grad-CAM**: For explainable AI.
    - **Google Translate**: For multilingual support.
    
    ### Data Sources:
    - Crop data from agricultural studies.
    - Plant disease images from PlantVillage dataset.
    
    ### Creators:
    Developed as a project to demonstrate AI in agriculture. For feedback or questions, contact the developer.
    
    ### Disclaimer:
    This app provides general advice. Always consult local agricultural experts for specific recommendations. Results are based on AI predictions and may not be 100% accurate.
    """, lang), unsafe_allow_html=True)
    st.image("https://via.placeholder.com/600x300?text=AI+in+Agriculture", caption=translate_text("Empowering farmers with AI", lang))

# Footer
st.markdown("---")
st.markdown(translate_text("🌾 **AI Farmer Advisor** - Helping farmers grow smarter! | For support, contact local agricultural services.", lang))
st.markdown("© 2025 AI Agriculture Project")
