import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy as scipy_entropy
import joblib
from keras.models import load_model
import base64
from io import BytesIO

# Fungsi-fungsi pre-processing, segmentasi, dan ekstraksi fitur
def preproc(input_img):
    image = Image.open(input_img)
    img_array = np.array(image)
    resized_img = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    filtered_image = cv2.bilateralFilter(gray_img, 3, 4, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(filtered_image, cv2.MORPH_BLACKHAT, kernel)
    ret, mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    kernel_morph = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel_morph, iterations=1)
    erosion = cv2.erode(dilation, kernel_morph, iterations=1)
    outputDR = cv2.inpaint(resized_img, erosion, 6, cv2.INPAINT_TELEA)
    return resized_img, gray_img, filtered_image, blackhat, mask, erosion, outputDR

def segmentationmask(outputDR, model):
    predictmask = model.predict(np.expand_dims(outputDR, axis=0))
    pred_mask = np.squeeze(predictmask)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)*255
    return pred_mask

def cropmask(outputDR, pred_mask):
    if pred_mask.ndim == 3 and pred_mask.shape[2] == 1:
        pred_mask = pred_mask[:, :, 0]
    _, binary_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask_3c = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(outputDR, binary_mask_3c)
    return masked_image

def overlay(outputDR, pred_mask):
    if pred_mask.ndim == 3 and pred_mask.shape[2] == 1:
        pred_mask = pred_mask[:, :, 0]
    colored_mask = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(outputDR, 0.7, colored_mask, 0.3, 0)
    return overlay

def glcm_features(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    contrast = {}
    energy = {}
    homogeneity = {}
    entropy = {}

    for angle in angles:
        glcm = graycomatrix(gray_img, distances=distances, angles=[angle], levels=256, symmetric=True, normed=True)
        contrast[f"Contrast_{int(np.degrees(angle))}"] = graycoprops(glcm, 'contrast')[0, 0]
        energy[f"Energy_{int(np.degrees(angle))}"] = graycoprops(glcm, 'energy')[0, 0]
        homogeneity[f"Homogeneity_{int(np.degrees(angle))}"] = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy[f"Entropy_{int(np.degrees(angle))}"] = scipy_entropy(glcm.ravel(), base=2)
    return contrast, energy, homogeneity, entropy

def abcd_feature(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None

    lesion_contour = max(contours, key=cv2.contourArea)
    if len(lesion_contour) >= 5:
        ellipse = cv2.fitEllipse(lesion_contour)
        (x, y), (MA, ma), angle = ellipse
        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
        lesion_contour = lesion_contour.astype(np.float32).reshape(-1, 1, 2)
        rotated_contour = cv2.transform(lesion_contour, rotation_matrix).reshape(-1, 2)
        x_coords = rotated_contour[:, 0]
        median_x = np.median(x_coords)
        left_contour = rotated_contour[x_coords <= median_x]
        right_contour = rotated_contour[x_coords > median_x]

        if len(left_contour) > 2:
            left_contour = np.vstack([left_contour, left_contour[0]])
            left_area = cv2.contourArea(left_contour)
        else:
            left_area = 0

        if len(right_contour) > 2:
            right_contour = np.vstack([right_contour, right_contour[0]])
            right_area = cv2.contourArea(right_contour)
        else:
            right_area = 0

        if left_area + right_area > 0:
            asymmetry = abs(left_area - right_area) / max(left_area, right_area)
        else:
            asymmetry = np.nan
    else:
        asymmetry = np.nan

    edges = cv2.Canny(gray_img, 100, 200)
    border = np.count_nonzero(edges)
    colors = cv2.mean(image)[:3]
    rect = cv2.minAreaRect(lesion_contour)
    width, height = rect[1]
    diameter = max(width, height)
    return asymmetry, border, colors[2], colors[1], colors[0], diameter

# Memuat model segmentasi dan klasifikasi
model = load_model(r"D:\ITS\Semester 8\Tugas Akhir\CODE\unet_segmentasi.h5")
svm_model = joblib.load(r"D:\ITS\Semester 8\Tugas Akhir\CODE\svm_BalancingPCC_ASDIA_21.pkl")


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


logo_path = r"C:\Users\ASUS\Downloads\Input Dataset (31).png"
logo = Image.open(logo_path)
logo_base64 = image_to_base64(logo)

st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <h3 style='text-align: center; font-weight: bold;'>
        SKIN CANCER CLASSIFICATION FROM DERMATOSCOPY IMAGES BASED ON CONVOLUTIONAL NEURAL NETWORK U-NET AND SUPPORT VECTOR MACHINE
    </h3>
""", unsafe_allow_html=True)

with st.expander(":clipboard: See abstract"):
    st.write("""
        <p style='text-align: justify;'>
                The Global Burden of Cancer Study (Globocan) by the World Health 
        Organization (WHO) reported that the number of cancer cases in Indonesia in 2020 
        reached 396,914, with total cancer-related deaths reaching 234,511 cases. Skin 
        cancer accounted for 5.9 - 7.8% of the total cancer cases. The cure rate can increase 
        up to 90% with early detection, but early detection is considered complex and 
        subjective, often leading to delayed skin cancer diagnosis. Consequently, a 
        Computer-Aided Diagnostic (CAD) system, designed to enhance diagnostic 
        accuracy, has been developed. Automated diagnosis of dermatoscopic images faces 
        challenges due to complex variations in appearance. Hence, pre-processing and 
        feature extraction are necessary to overcome these challenges. This study 
        implements a Convolutional Neural Network (CNN) with U-Net architecture and 
        Support Vector Machine (SVM) for the detection and classification of skin lesions 
        in dermoscopic images. Before entering the classification process, feature 
        extraction will be conducted using the ABCD and GLCM methods. This research 
        aims to improve the accuracy and effectiveness of skin cancer diagnosis by utilizing 
        U-Net for segmentation and SVM for classification. The dermoscopic image 
        analysis approach has the potential to enhance patient care quality and reduce the 
        risk of delayed diagnoses.
         </p>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([":desktop_computer: Main", ":hammer_and_wrench: Pre-Processing", ":petri_dish: Segmentation", ":1234: Feature Extraction", ":white_check_mark: Classification"])

with tab1:
    method_img = Image.open(r"D:\ITS\Semester 8\Tugas Akhir\else\Input Dataset (4).png")
    with st.expander(":mag: Metodology"):
        st.image(method_img)
        
    # Upload image
    input_img = st.file_uploader("Please Upload an Image", type=["png", "jpg"])
    if input_img is not None:
        progress_bar = st.progress(0)
        for perc_completed in range(101):
            time.sleep(0.05)
            progress_bar.progress(perc_completed)
        st.image(input_img)
        st.success("Photo uploaded successfully!")
    
    # Run button
    run_btn = st.button(label="Start", key="run_all")
    if run_btn: 
        if input_img is not None: 
            #bagian prepros
            resized_img, gray_img, filtered_image, blackhat, mask, erosion, outputDR = preproc(input_img)
    
            #bagian segmentasi 
            hasilsegmentasi = segmentationmask(outputDR, model)
            overlay = overlay(outputDR, hasilsegmentasi)
            hasilcrop = cropmask(outputDR, hasilsegmentasi)
            
            #bagian ekstraksi fitur 
            #glcm 
            contrast, energy, homogeneity, entropy = glcm_features(outputDR)
            data_glcm = {
                'Feature': ['Contrast', 'Energy', 'Homogeneity', 'Entropy'],
                '0 \u00b0': [contrast['Contrast_0'], energy['Energy_0'], homogeneity['Homogeneity_0'], entropy['Entropy_0']],
                '45 \u00b0': [contrast['Contrast_45'], energy['Energy_45'], homogeneity['Homogeneity_45'], entropy['Entropy_45']],
                '90 \u00b0': [contrast['Contrast_90'], energy['Energy_90'], homogeneity['Homogeneity_90'], entropy['Entropy_90']],
                '135 \u00b0': [contrast['Contrast_135'], energy['Energy_135'], homogeneity['Homogeneity_135'], entropy['Entropy_135']]
            }
            df_glcm = pd.DataFrame(data_glcm)
            #abcd 
            asymmetry, border, red_mean, green_mean, blue_mean, diameter = abcd_feature(hasilcrop)
            data_abcd = {
                'Asymmetry': [asymmetry],
                'Border': [border],
                'Color Red': [red_mean],
                'Color Green': [green_mean],
                'Color Blue': [blue_mean],
                'Diameter': [diameter]
            }
            df_abcd = pd.DataFrame(data_abcd)
            #total 
            combined_data = {
                'Homogeneity_45': [homogeneity['Homogeneity_45']],
                'Homogeneity_0': [homogeneity['Homogeneity_0']],
                'Homogeneity_135': [homogeneity['Homogeneity_135']],
                'Energy_45': [energy['Energy_45']],
                'Energy_0': [energy['Energy_0']],
                'Energy_135': [energy['Energy_135']],
                'Energy_90': [energy['Energy_90']],
                'Homogeneity_90': [homogeneity['Homogeneity_90']],
                'Green': [green_mean],
                'Entropy_45': [entropy['Entropy_45']],
                'Border': [border],
                'Entropy_0': [entropy['Entropy_0']],
                'Entropy_135': [entropy['Entropy_135']],
                'Entropy_90': [entropy['Entropy_90']],
                'Blue': [blue_mean],
                'Contrast_0': [contrast['Contrast_0']],
                'Contrast_45': [contrast['Contrast_45']],
                'Contrast_135': [contrast['Contrast_135']],
                'Contrast_90': [contrast['Contrast_90']],
                'Red': [red_mean],
                'Diameter': [diameter]
                
            }
            df_combined = pd.DataFrame(combined_data)
            
            #klasifikasi 
            prediction = svm_model.predict(df_combined)
            label_mapping = {
                0: 'Actinic Keratosis',
                1: 'Basal Cell Carcinoma',
                2: 'Benign Keratosis',
                3: 'Dermatofibroma',
                4: 'Melanoma',
                5: 'Melanocytic Nevus',
                6: 'Vascular Lesions'
            }
            predicted_label = label_mapping.get(prediction[0], "Unknown")
            
            #tampilan 
            col1, col2 = st.columns(2)
            col1.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Preprocessing Result</h2>", unsafe_allow_html=True)
            col1.image(outputDR)
            col2.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Segmentation Result</h2>", unsafe_allow_html=True)
            col2.image(hasilsegmentasi)
            st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Feature Extraction Result</h2>", unsafe_allow_html=True)
            st.table(df_combined)
            st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Predicted Label</h2>", unsafe_allow_html=True)
            st.write(predicted_label)
        else : 
            st.warning("Please upload an image first!")
            
            
            

with tab2:
    st.header("Pre-Processing")
    preprosimg = Image.open(r"D:\ITS\Semester 8\Tugas Akhir\else\Input Dataset.png")
    with st.expander(":mag: Step By Step"):
        st.image(preprosimg)
    run_prepros = st.button(label="Start Pre-Processing", key="run_preprosaja")
    if run_prepros:
        if input_img is not None:
            resized_img, gray_img, filtered_image, blackhat, mask, erosion, outputDR = preproc(input_img)
            st.session_state['outputDR'] = outputDR
            col1, col2, col3 = st.columns(3)
            col1.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Image Resize</h2>", unsafe_allow_html=True)
            col1.image(resized_img)
            col2.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Image Enhancement</h2>", unsafe_allow_html=True)
            col2.image(gray_img)
            col3.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Image Restoration</h2>", unsafe_allow_html=True)
            col3.image(filtered_image)
            st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Noise Removal</h2>", unsafe_allow_html=True)
            col1_noise, col2_noise, col3_noise = st.columns(3)
            col1_noise.caption("Black Hat Filter")
            col1_noise.image(blackhat)
            col2_noise.caption("Binary thresholding")
            col2_noise.image(mask)
            col3_noise.caption("Closing Morphological")
            col3_noise.image(erosion)
            st.markdown("Output")
            st.image(outputDR, caption='Processed Image', use_column_width=True)
            
            # Progress bar
            progress_bar = st.progress(100)
            st.success("Pre-processing completed successfully!")
        else:
            st.warning("Please upload an image first!")

with tab3:
    st.header("Segmentation")
    method_segmen = Image.open(r"D:\ITS\Semester 8\Tugas Akhir\else\U-Net Amanda - bab 3.png")
    with st.expander(":mag: U-Net Architecture"):
        st.image(method_segmen, caption='U-Net Architecture Illustration')
    run_segmen = st.button(label="Start Segmentation", key="run_segmenaja")
    if run_segmen:
        if 'outputDR' in st.session_state:
            outputDR = st.session_state['outputDR']
            col1, col2, col3 = st.columns(3)
            col1.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Image Segmentation</h2>", unsafe_allow_html=True)
            hasilsegmentasi = segmentationmask(outputDR, model)
            col1.image(hasilsegmentasi)
            col2.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Overlay Image</h2>", unsafe_allow_html=True)
            overlay = overlay(outputDR, hasilsegmentasi)
            col2.image(overlay)
            col3.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Crop Mask Image</h2>", unsafe_allow_html=True)
            hasilcrop = cropmask(outputDR, hasilsegmentasi)
            st.session_state['hasilcrop'] = hasilcrop
            col3.image(hasilcrop)
    
            # Progress bar
            progress_bar = st.progress(100)
            st.success("Segmentation completed successfully!")
        else:
            st.warning("Please assign pre-processing first!")

with tab4:
    st.header("Feature Extraction")
    with st.expander(":mag: Explanation"):
        st.write("""
                <h2 style='font-size: 18px; font-weight: bold;'>Feature Extraction with the ABCD Method</h2>
                A : Asymmetry 
                Measures the difference in shape between two halves of the image divided by a central line. Asymmetric lesions are more suspicious.
                B : Border 
                Measures the number of pixels detected as edges in the image. The higher this value, the more irregular the borders of the skin lesion.
                C : Color 
                Measures the intensity for each color channel, reflecting the color variation within the skin lesion. This value is used to assess how varied the colors are within the lesion, which is an important indicator in dermatological diagnosis.
                D : Diameter 
                Measures the size of the lesion. Lesions larger than 6 mm are more suspicious.
                
                <h2 style='font-size: 18px; font-weight: bold;'>Feature Extraction with the GLCM Method</h2>
                The Gray Level Co-occurrence Matrix (GLCM) is a statistical method used to extract texture features from images based on the spatial distribution of pixel intensities. 
                Contrast : Measures the intensity difference between a pixel and its neighbor over the entire image.
                Energy : Measures the homogeneity of the texture in the image. High energy indicates a homogeneous image.
                Entropy : Measures the uncertainty or randomness in the intensity distribution. High entropy indicates a complex image.
                Homogeneity : Measures how close the distribution of elements in the GLCM is to the diagonal. High homogeneity indicates a uniform intensity distribution.
                 """)

    col_glcm, col_abcd = st.columns(2)
    run_ekstraksi = st.button(label="Start Feature Extraction", key="run_extraction")
    if run_ekstraksi:
        # GLCM
        st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>GLCM</h2>", unsafe_allow_html=True)
        if 'outputDR' in st.session_state:
            outputDR = st.session_state['outputDR']
            contrast, energy, homogeneity, entropy = glcm_features(outputDR)
            data_glcm = {
                'Feature': ['Contrast', 'Energy', 'Homogeneity', 'Entropy'],
                '0 \u00b0': [contrast['Contrast_0'], energy['Energy_0'], homogeneity['Homogeneity_0'], entropy['Entropy_0']],
                '45 \u00b0': [contrast['Contrast_45'], energy['Energy_45'], homogeneity['Homogeneity_45'], entropy['Entropy_45']],
                '90 \u00b0': [contrast['Contrast_90'], energy['Energy_90'], homogeneity['Homogeneity_90'], entropy['Entropy_90']],
                '135 \u00b0': [contrast['Contrast_135'], energy['Energy_135'], homogeneity['Homogeneity_135'], entropy['Entropy_135']]
            }
            df_glcm = pd.DataFrame(data_glcm)
            st.table(df_glcm)
        else:
            st.warning("Please assign pre-processing first!")
        
        st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>ABCD</h2>", unsafe_allow_html=True)
        if 'hasilcrop' in st.session_state:
            hasilcrop = st.session_state['hasilcrop']
            asymmetry, border, red_mean, green_mean, blue_mean, diameter = abcd_feature(hasilcrop)
            data_abcd = {
                'Asymmetry': [asymmetry],
                'Border': [border],
                'Color Red': [red_mean],
                'Color Green': [green_mean],
                'Color Blue': [blue_mean],
                'Diameter': [diameter]
            }
            df_abcd = pd.DataFrame(data_abcd)
            st.table(df_abcd)
        
            combined_data = {
                'Homogeneity_45': [homogeneity['Homogeneity_45']],
                'Homogeneity_0': [homogeneity['Homogeneity_0']],
                'Homogeneity_135': [homogeneity['Homogeneity_135']],
                'Energy_45': [energy['Energy_45']],
                'Energy_0': [energy['Energy_0']],
                'Energy_135': [energy['Energy_135']],
                'Energy_90': [energy['Energy_90']],
                'Homogeneity_90': [homogeneity['Homogeneity_90']],
                'Green': [green_mean],
                'Entropy_45': [entropy['Entropy_45']],
                'Border': [border],
                'Entropy_0': [entropy['Entropy_0']],
                'Entropy_135': [entropy['Entropy_135']],
                'Entropy_90': [entropy['Entropy_90']],
                'Blue': [blue_mean],
                'Contrast_0': [contrast['Contrast_0']],
                'Contrast_45': [contrast['Contrast_45']],
                'Contrast_135': [contrast['Contrast_135']],
                'Contrast_90': [contrast['Contrast_90']],
                'Red': [red_mean],
                'Diameter': [diameter],
                #'Asymmetry': [asymmetry]
            }
            df_combined = pd.DataFrame(combined_data)
            
            #st.markdown("<h2 style='font-size: 18px; font-weight: bold;'>Combined Features</h2>", unsafe_allow_html=True)
            #st.table(df_combined)
            st.session_state['df_combined'] = df_combined
        else:
            st.warning("Please assign segmentation first!")

with tab5:
    st.header("Classification")
    with st.expander(":mag: Explanation"):
        st.write("""
                 To classify lesions, an SVM with an RBF kernel was used, with C = 1000 and gamma = 0.1.
                 """)
    run_klasifikasi = st.button(label="Start Classification", key="run_classification")
    if run_klasifikasi:
        if 'df_combined' in st.session_state:
            df_combined = st.session_state['df_combined']
            prediction = svm_model.predict(df_combined)
            label_mapping = {
                0: 'Actinic Keratosis',
                1: 'Basal Cell Carcinoma',
                2: 'Benign Keratosis',
                3: 'Dermatofibroma',
                4: 'Melanoma',
                5: 'Melanocytic Nevus',
                6: 'Vascular Lesions'
            }
            predicted_label = label_mapping.get(prediction[0], "Unknown")
            st.write(f"Predicted Label: {predicted_label}")
           
        else:
            st.warning("Please run feature extraction first!")

st.markdown("""
    <div style='text-align: right; font-size: smaller;'>
        Amanda Sharon Purwanti Junior 5023201044
    </div>
    <div style='text-align: right; font-size: smaller;'>
        Dosen Pembimbing :
    </div>
    <div style='text-align: right; font-size: smaller;'>
        Prof. Dr. Ir. Mohammad Nuh, DEA.    195906171984031002
    </div>
    <div style='text-align: right; font-size: smaller;'>
		Nada Fitrieyatul Hikmah, S.T., M.T. 199001072018032001
    </div>
""", unsafe_allow_html=True)
