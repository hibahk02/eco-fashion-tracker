
import streamlit as st
import pandas as pd
import pickle
import os
import datetime
import pydeck as pdk
import numpy as np

# Emojis
emoji_home = "🏠"
emoji_score = "🧮"
emoji_recommend = "🧭"
emoji_dashboard = "📊"
emoji_map = "🌍"
emoji_green = "🌱"
emoji_leaf = "🌿"
emoji_medal = "🏅"
emoji_download = "📥"
emoji_warning = "🚨"

st.set_page_config(page_title="Eco Fashion Tracker " + emoji_green, layout="wide")

# Load data (must exist next to app.py)
df = pd.read_csv("sustainable_with_rating.csv")

# pre-calc averages per material
material_avg_values = df.groupby('Material_Type')[[
    'Carbon_Footprint_KG', 'Water_Usage_KG', 'Waste_Production_KG'
]].mean().to_dict(orient='index')

# fallback average for missing materials
fallback_avg = df[['Carbon_Footprint_KG', 'Water_Usage_KG', 'Waste_Production_KG']].mean().to_dict()

# Prepare material list and "Other"
all_materials = sorted(df['Material_Type'].dropna().unique().tolist())
if "Other" not in all_materials:
    all_materials.append("Other")

USER_RESULTS_FILE = 'user_results.csv'

def save_user_results(product_name, brand, materials, country, recycling, eco_score, score_value):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    materials_str = ", ".join(materials)
    if os.path.exists(USER_RESULTS_FILE):
        results_df = pd.read_csv(USER_RESULTS_FILE)
    else:
        results_df = pd.DataFrame(columns=[
            "Product_Name", "Brand_Name", "Material_Type", "Country",
            "Recycling_Programs", "Eco_Score", "Numeric_Score", "Timestamp"
        ])
    new_result = pd.DataFrame({
        "Product_Name": [product_name],
        "Brand_Name": [brand],
        "Material_Type": [materials_str],
        "Country": [country],
        "Recycling_Programs": [recycling],
        "Eco_Score": [eco_score],
        "Numeric_Score": [round(score_value, 4)],
        "Timestamp": [timestamp]
    })
    results_df = pd.concat([results_df, new_result], ignore_index=True)
    results_df.to_csv(USER_RESULTS_FILE, index=False)

# ---- Artifacts (model + optional preprocessor) ----
@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    preprocessor = None
    try:
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        pass
    return best_model, preprocessor

best_model, preprocessor = load_artifacts()

# === Schema alignment helpers (must be above any predict calls) ===
def _find_expected_feature_list(model):
    if hasattr(model, "named_steps"):  # Pipeline
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):  # Single estimator
        return list(model.feature_names_in_)
    return None

def _align_to_expected_schema(model, df_in):
    expected = _find_expected_feature_list(model)
    if not expected:
        return df_in.copy()
    aligned = df_in.copy()
    # If model expects an encoded recycling flag, create it
    if "Recycling_Programs_Encoded" in expected and "Recycling_Programs_Encoded" not in aligned.columns:
        if "Recycling_Programs" in aligned.columns:
            aligned["Recycling_Programs_Encoded"] = (
                aligned["Recycling_Programs"].astype(str).str.lower() == "yes"
            ).astype(int)
        else:
            aligned["Recycling_Programs_Encoded"] = 0
    # Fill any missing expected columns with neutral defaults
    missing = [c for c in expected if c not in aligned.columns]
    for c in missing:
        if any(tok in c.lower() for tok in [
            "type","name","id","country","certification","trend","material",
            "brand","program","eco","recycling","product","market"
        ]):
            aligned[c] = ""
        else:
            aligned[c] = 0.0
    # Exact training order
    aligned = aligned.reindex(columns=expected)
    return aligned

def predict_aligned(model, preproc, df_in):
    # Align columns; then transform if we have a separate preprocessor
    aligned = _align_to_expected_schema(model, df_in)
    if preproc is not None:
        X = preproc.transform(aligned)
        return model.predict(X)
    else:
        # If your model is a Pipeline that already includes preprocessing, this still works.
        return model.predict(aligned)
# === end helpers ===

# Sidebar menu
menu = st.sidebar.radio("Menu", [
    f"{emoji_home} Home",
    f"{emoji_score} Score a Product",
    f"{emoji_recommend} Recommendations",
    f"{emoji_dashboard} Dashboard",
    f"{emoji_map} Map"
])

# Home Page
if menu == f"{emoji_home} Home":
    st.title("Eco Fashion Tracker " + emoji_green)
    st.write(f'''
Welcome to **Eco Fashion Tracker**! 🌿

Measure, improve, and celebrate your sustainable fashion choices.

- {emoji_score} **Score a Product** to see its eco-score
- {emoji_recommend} **Recommendations** for better material choices
- {emoji_dashboard} **Dashboard** to track your progress
- {emoji_map} **Map** to visualize product origins globally

Let's make fashion greener together! 💚
''')

    st.header(f"{emoji_leaf} Understanding Eco-Scores")
    st.write('''
Here's what your Eco-Scores mean:

- **🅰️ Eco Score A (Excellent):** Outstanding sustainability performance
- **🅱️ Eco Score B (Good):** Good sustainability standards, some room for improvement
- **🅲 Eco Score C (Average):** Moderate performance, improvements recommended
- **🅳 Eco Score D (Poor):** Low sustainability performance. Consider changes
''')

# Score a Product Page
if menu == f"{emoji_score} Score a Product":
    st.header("Enter Product Details")

    st.write("ℹ️ **Note:** You can usually find the material composition and country of origin on the item's tag.")
    st.write("♻️ **Second-hand / recycled items** help reduce waste and resource use.")

    product_name = st.text_input("Product Name", placeholder="E.g., Summer Dress, Winter Jacket")
    brand = st.text_input("Brand Name", placeholder="E.g., Zara, H&M")
    materials_selected = st.multiselect("Material Type(s)", all_materials)

    # Countries dropdown
    countries = sorted(df['Country'].dropna().unique().tolist()) if 'Country' in df.columns else [
        "United Kingdom","United States","China","India","Bangladesh","Vietnam","Turkey","Italy","Spain","France","Germany","Netherlands","Portugal","Pakistan","Cambodia","Mexico","Brazil","Indonesia","Thailand","Sri Lanka","Japan","South Korea"
    ]
    country = st.selectbox("Country of Origin", countries)
    second_hand = st.checkbox("Is the product second-hand (recycled)?")
    recycling = "Yes" if second_hand else "No"
    if second_hand:
        st.info("Recycled products get a sustainability boost! 🌱")

    debug_mode = st.checkbox("Show debug info")

    if st.button("Get Eco-Score"):
        if not product_name or not brand or not country or not materials_selected:
            st.warning(f"{emoji_warning} Please fill in all the required fields!")
        else:
            selected_values = [material_avg_values.get(m, fallback_avg) for m in materials_selected]
            carbon = sum(v["Carbon_Footprint_KG"] for v in selected_values) / len(selected_values)
            water  = sum(v["Water_Usage_KG"] for v in selected_values) / len(selected_values)
            waste  = sum(v["Waste_Production_KG"] for v in selected_values) / len(selected_values)

            if recycling == "Yes":
                carbon *= 0.8; water *= 0.8; waste *= 0.8

            material_type_for_model = (
                materials_selected[0] if materials_selected[0] != "Other"
                else df['Material_Type'].mode()[0]
            )

            input_df = pd.DataFrame({
                "Material_Type": [material_type_for_model],
                "Country": [country],
                "Recycling_Programs": [recycling],
                "Carbon_Footprint_KG": [carbon],
                "Water_Usage_KG": [water],
                "Waste_Production_KG": [waste]
            })

            if debug_mode:
                st.write("🔍 Raw model input", input_df)

            # Predict using aligned + (optional) preprocessed features
            pred_numeric = predict_aligned(best_model, preprocessor, input_df)[0]

            # Thresholds from a sample of the dataset
            sample_df = df.sample(500 if len(df) > 500 else len(df)).copy()
            sample_preds = predict_aligned(best_model, preprocessor, sample_df)

            q1 = np.percentile(sample_preds, 25)
            q2 = np.percentile(sample_preds, 50)
            q3 = np.percentile(sample_preds, 75)

            if debug_mode:
                st.write(f"Thresholds — Q3 (D): {q3:.2f}, Q2 (C): {q2:.2f}, Q1 (B): {q1:.2f}")

            if pred_numeric > q3:
                eco_score_category = 'D'
            elif pred_numeric > q2:
                eco_score_category = 'C'
            elif pred_numeric > q1:
                eco_score_category = 'B'
            else:
                eco_score_category = 'A'

            st.session_state.eco_score_category = eco_score_category
            st.success(f"Eco-Score Calculated! Your product's eco-score is: **{eco_score_category}**")
            save_user_results(product_name, brand, materials_selected, country, recycling, eco_score_category, float(pred_numeric))

            if eco_score_category in ['A', 'B']:
                st.balloons()
            else:
                st.snow()

# Recommendations Page
if menu == f"{emoji_recommend} Recommendations":
    eco_score_category = st.session_state.get("eco_score_category", None)
    if eco_score_category:
        st.header("Sustainable Alternatives")
        st.write("Based on your eco-score, here are some actionable recommendations:")
        if eco_score_category == 'A':
            st.markdown("- Keep up the great work! Choose products made from sustainable materials.")
            st.markdown("- Share what you’ve learned and support ethical brands.")
        elif eco_score_category == 'B':
            st.markdown("- Prefer higher recycled-content materials.")
            st.markdown("- Consider second-hand/vintage options to cut new production.")
        elif eco_score_category == 'C':
            st.markdown("- Switch towards organic cotton, hemp, or recycled polyester.")
            st.markdown("- Avoid high-impact synthetics when possible.")
        elif eco_score_category == 'D':
            st.markdown("- Learn about sustainability impacts and support circular programs.")
            st.markdown("- Prefer brands with certifications (Fair Trade, GOTS, B-Corp).")
    else:
        st.warning("Please calculate the eco-score first in the 'Score a Product' section.")

    st.header("Material Recommendations")
    material_impact = df.groupby('Material_Type')[
        ['Carbon_Footprint_KG', 'Water_Usage_KG', 'Waste_Production_KG']
    ].mean().sort_values('Carbon_Footprint_KG')
    st.dataframe(material_impact)

# Dashboard Page
if menu == f"{emoji_dashboard} Dashboard":
    st.header("Your Sustainability Progress")
    if os.path.exists(USER_RESULTS_FILE):
        if "results_df" not in st.session_state:
            _udf = pd.read_csv(USER_RESULTS_FILE)
            _udf.columns = _udf.columns.str.strip()
            _udf.reset_index(drop=True, inplace=True)
            st.session_state.results_df = _udf
        else:
            _udf = st.session_state.results_df
        if not _udf.empty:
            st.write("Your saved eco-score results:")
            st.dataframe(_udf)
            csv = _udf.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download your results as CSV",
                data=csv,
                file_name='eco_score_results.csv',
                mime='text/csv',
            )
            total = len(_udf)
            count_a = (_udf['Eco_Score'] == 'A').sum()
            pct_a = (count_a / total) * 100 if total else 0
            st.metric("Total Products", total)
            st.metric("A Grade Products", f"{count_a} ({pct_a:.1f}%)")
        else:
            st.info("No results yet. Go to 'Score a Product' to get started!")
    else:
        st.info("No results yet. Go to 'Score a Product' to get started!")

# Map Page
if menu == f"{emoji_map} Map":
    st.header("Global Map of Product Origins")
    if os.path.exists(USER_RESULTS_FILE):
        results_df = pd.read_csv(USER_RESULTS_FILE)
        if not results_df.empty:
            # A minimal scatter plot using lat/lon columns if present,
            # else skip map to avoid noisy errors in first run.
            if {'Country'}.issubset(results_df.columns):
                # Quick approximate coords (use your full dict later if desired)
                country_coords = {
                    'United Kingdom': [55.378051, -3.435973],
                    'United States': [37.09024, -95.712891],
                    'China': [35.86166, 104.195397],
                    'India': [20.593684, 78.96288],
                }
                results_df['lat'] = results_df['Country'].map(lambda x: country_coords.get(x, [0,0])[0])
                results_df['lon'] = results_df['Country'].map(lambda x: country_coords.get(x, [0,0])[1])
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=results_df,
                            get_position='[lon, lat]',
                            get_radius=100000,
                            pickable=True,
                        ),
                    ],
                    tooltip={"text": "Country: {Country}\nProduct: {Product_Name}"}
                ))
            else:
                st.info("No country data yet — score some products first!")
        else:
            st.info("No data to display yet. Score a product first!")
    else:
        st.info("No data to display yet. Score a product first!")
