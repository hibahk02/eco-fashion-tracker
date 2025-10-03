
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

# Load data
df = pd.read_csv("sustainable_with_rating.csv")

# pre-calculate average impact values per material
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

# save user results function
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

# Load model pipeline
# Load model + preprocessor
@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return best_model, preprocessor

best_model, preprocessor = load_artifacts()

# === Schema alignment helpers (must be above any predict calls) ===
def _find_expected_feature_list(model):
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def _align_to_expected_schema(model, df_in):
    import pandas as pd
    expected = _find_expected_feature_list(model)
    if not expected:
        return df_in.copy()
    aligned = df_in.copy()

    # If model expects an encoded recycling flag, create it from the string version
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

    # Ensure exact training order
    aligned = aligned.reindex(columns=expected)
    return aligned

def predict_aligned(model, preproc, df_in):
    # Align columns, then transform with preprocessor, then predict
    aligned = _align_to_expected_schema(model, df_in)
    X = preproc.transform(aligned)  # dense or sparse, both are fine
    return model.predict(X)



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
    st.write("♻️ **Second-hand / recycled items** are products that have been previously owned or are made from recycled materials, helping to reduce waste and resource consumption.")

    product_name = st.text_input("Product Name", placeholder="E.g., Summer Dress, Winter Jacket")
    brand = st.text_input("Brand Name", placeholder="E.g., Zara, H&M")
    materials_selected = st.multiselect("Material Type(s)", all_materials)
    # Define list of countries
    countries = [
        'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
        'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
        'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia',
        'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso',
        'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
        'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Congo-Brazzaville)', 'Costa Rica',
        'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Democratic Republic of the Congo', 'Denmark',
        'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
        'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland',
        'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada',
        'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland',
        'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan',
        'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
        'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
        'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands',
        'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia',
        'Montenegro', 'Morocco', 'Mozambique', 'Myanmar (Burma)', 'Namibia', 'Nauru', 'Nepal',
        'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia',
        'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay',
        'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda',
        'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
        'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone',
        'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea',
        'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria',
        'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago',
        'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates',
        'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela',
        'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
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
            water = sum(v["Water_Usage_KG"] for v in selected_values) / len(selected_values)
            waste = sum(v["Waste_Production_KG"] for v in selected_values) / len(selected_values)

            if recycling == "Yes":
                carbon *= 0.8
                water *= 0.8
                waste *= 0.8

            material_type_for_model = materials_selected[0] if materials_selected[0] != "Other" else df['Material_Type'].mode()[0]

            input_df = pd.DataFrame({
                "Material_Type": [material_type_for_model],
                "Country": [country],
                "Recycling_Programs": [recycling],
                "Carbon_Footprint_KG": [carbon],
                "Water_Usage_KG": [water],
                "Waste_Production_KG": [waste]
            })

            if debug_mode:
                st.write("🔍 Debug: Model input DataFrame")
                st.write(input_df)
                # Optional: show transformed shape
                _aligned_preview = _align_to_expected_schema(best_model, input_df)
                Xt_prev = preprocessor.transform(_aligned_preview)
                try:
                    import scipy.sparse as sp
                    if sp.issparse(Xt_prev):
                        st.write("Transformed X (sparse) shape:", Xt_prev.shape)
                    else:
                        st.write("Transformed X (dense) shape:", Xt_prev.shape)
                except Exception:
                    pass

            # Predict on preprocessed, aligned features
            pred_numeric = predict_aligned(best_model, preprocessor, input_df)[0]

            # Compute thresholds from a sample of the dataset (also preprocessed)
            sample_df = df.sample(500 if len(df) > 500 else len(df)).copy()
            sample_preds = predict_aligned(best_model, preprocessor, sample_df)

            q1 = np.percentile(sample_preds, 25)
            q2 = np.percentile(sample_preds, 50)
            q3 = np.percentile(sample_preds, 75)

            if debug_mode:
                st.write(f"Thresholds: Q3 (D): {q3:.2f}, Q2 (C): {q2:.2f}, Q1 (B): {q1:.2f}")

            if pred_numeric > q3:
                eco_score_category = 'D'
            elif pred_numeric > q2:
                eco_score_category = 'C'
            elif pred_numeric > q1:
                eco_score_category = 'B'
            else:
                eco_score_category = 'A'

            # store eco_score_category in session state
            st.session_state.eco_score_category = eco_score_category

            st.success(f"Eco-Score Calculated! Your product's eco-score is: **{eco_score_category}**")
            save_user_results(product_name, brand, materials_selected, country, recycling, eco_score_category, pred_numeric)
            st.info("Your result has been automatically saved!")

            if eco_score_category in ['A', 'B']:
                st.balloons()
            else:
                st.snow()



# Recommendations Page
if menu == f"{emoji_recommend} Recommendations":
    # retrieve eco_score_category from session state
    eco_score_category = st.session_state.get("eco_score_category", None)

    if eco_score_category:
        st.header("Sustainable Alternatives")
        st.write("Based on your product's eco-score, here are some actionable recommendations:")

        # conditional recommendations based on the eco score
        if eco_score_category == 'A':
            st.write("**For Eco Score A (Excellent):**")
            st.markdown("- **Keep up the good work!** Continue choosing products made from sustainable materials.")
            st.markdown("- **Share your knowledge:** Educate friends and family about the importance of sustainability in fashion.")
            st.markdown("- **Buy from ethical brands:** Support brands that prioritize sustainability and ethical practices.")
            st.markdown("- **Reuse and recycle:** Consider donating or repurposing clothes that you no longer need instead of throwing them away.")
            st.markdown("- **Reduce consumption:** Focus on buying high-quality, long-lasting items rather than frequently purchasing fast fashion.")

        elif eco_score_category == 'B':
            st.write("**For Eco Score B (Good):**")
            st.markdown("- **Optimize your purchases:** Look for products with a higher percentage of recycled or sustainable materials.")
            st.markdown("- **Buy second-hand:** Consider purchasing second-hand or vintage items to reduce demand for new production.")
            st.markdown("- **Invest in eco-friendly brands:** Choose brands that invest in eco-friendly processes and have clear sustainability certifications.")
            st.markdown("- **Care for your clothes:** Extend the lifespan of your garments by following proper care instructions and repairing items when needed.")
            st.markdown("- **Reduce waste:** Donate items you no longer use instead of sending them to landfills.")

        elif eco_score_category == 'C':
            st.write("**For Eco Score C (Average):**")
            st.markdown("- **Switch to sustainable fabrics:** Choose products made from organic cotton, hemp, or recycled polyester instead of conventional fabrics.")
            st.markdown("- **Support local and ethical brands:** Purchase from brands that source materials responsibly and ensure fair working conditions.")
            st.markdown("- **Avoid synthetic fibers:** Minimize synthetic fibers such as polyester, which contribute to microplastic pollution.")
            st.markdown("- **Consider eco-friendly laundry options:** Use a microfiber filter bag or a washing ball to prevent microplastics from polluting water.")
            st.markdown("- **Buy less, choose wisely:** Only purchase what you truly need, and opt for versatile pieces that can be worn multiple ways.")

        elif eco_score_category == 'D':
            st.write("**For Eco Score D (Poor):**")
            st.markdown("- **Educate yourself:** Learn more about the impact of your clothing choices on the environment and human rights.")
            st.markdown("- **Support circular fashion:** Look for brands that offer take-back programs or clothing repair services to reduce waste.")
            st.markdown("- **Switch to eco-friendly brands:** Look for certifications like Fair Trade, GOTS (Global Organic Textile Standard), or B Corp when purchasing clothing.")
            st.markdown("- **Repurpose and upcycle:** Get creative with repurposing old clothing instead of discarding it. Upcycling is a great way to reduce waste.")
            st.markdown("- **Reduce frequency of purchases:** Slow down your consumption by focusing on quality, durable clothing over fast fashion.")

        else:
            st.warning("No eco-score found! Please try again.")
    else:
        st.warning("Please calculate the eco-score first in the 'Score a Product' section.")

    st.header("Material Recommendations")
    st.write("Different materials have varying impacts on the environment. Materials like organic cotton, recycled polyester, and hemp generally have lower carbon footprints and water usage compared to conventional options.")
    st.write("🌿 **Better materials** reduce CO2 emissions, water consumption, and waste — making them more eco-friendly choices.")
    st.write("Consider opting for materials that are responsibly sourced or recycled whenever possible!")

    material_impact = df.groupby('Material_Type')[[
        'Carbon_Footprint_KG', 'Water_Usage_KG', 'Waste_Production_KG'
    ]].mean().sort_values('Carbon_Footprint_KG')
    st.dataframe(material_impact)

    if os.path.exists(USER_RESULTS_FILE):
        results_df = pd.read_csv(USER_RESULTS_FILE)

        if not results_df.empty:
            top_materials = ["Hemp", "Tencel", "Bamboo fabric"]

            st.write("Based on sustainable practices, consider these top eco-friendly materials:")
            for material in top_materials:
                st.markdown(f"- {material}")

            common_material = results_df['Material_Type'].value_counts().idxmax()
            st.write(f"You've mostly used **{common_material}**. Consider trying:")
            for material in top_materials:
                if material != common_material:
                    st.markdown(f"- {material}")

            count_a = (results_df['Eco_Score'] == 'A').sum()
            if count_a >= 5:
                st.success(f"{emoji_medal} Achievement unlocked: Scored 'A' grade {count_a} times!")

        else:
            st.info("No user data yet! Use 'Score a Product' to get personalized recommendations.")
    else:
        st.info("No user data yet! Use 'Score a Product' to get personalized recommendations.")

# Dashboard Page
if menu == f"{emoji_dashboard} Dashboard":
    st.header("Your Sustainability Progress")

    # Load and prep data
    if os.path.exists(USER_RESULTS_FILE):
        if "results_df" not in st.session_state:
            df = pd.read_csv(USER_RESULTS_FILE)
            df.columns = df.columns.str.strip()
            df.reset_index(drop=True, inplace=True)
            st.session_state.results_df = df
        else:
            df = st.session_state.results_df

        if not df.empty:
            # TABLE: only one main display, always at the top
            st.write("Your saved eco-score results:")
            st.dataframe(df)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download your results as CSV",
                data=csv,
                file_name='eco_score_results.csv',
                mime='text/csv',
            )

            # Metrics
            total_entries = len(df)
            count_a = (df['Eco_Score'] == 'A').sum()
            percentage_a = (count_a / total_entries) * 100
            st.metric("Total Products", total_entries)
            st.metric("A Grade Products", f"{count_a} ({percentage_a:.1f}%)")

            # Row deletion UI
            selected_row = st.selectbox(
                "Select a row to delete (matches left column above):",
                options=df.index,
                format_func=lambda i: f"{i}: {df.loc[i, 'Product_Name']} | {df.loc[i, 'Eco_Score']}"
            )

            confirm_delete = st.checkbox("✅ Confirm delete selected row")
            if st.button("Delete Selected Entry 🗑️"):
                if confirm_delete:
                    df = df.drop(index=selected_row).reset_index(drop=True)
                    df.to_csv(USER_RESULTS_FILE, index=False)
                    st.session_state.results_df = df  # update session copy
                    st.success("✅ Row deleted and table updated.")
                else:
                    st.warning("⚠️ Please confirm deletion.")

            # Delete all
            confirm_all = st.checkbox("✅ Confirm delete ALL entries")
            if st.button("Delete All Results 🗑️"):
                if confirm_all:
                    os.remove(USER_RESULTS_FILE)
                    st.session_state.pop("results_df", None)
                    st.success("✅ All results deleted.")
                else:
                    st.warning("Please confirm deletion of all entries.")
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
            # country coordinates
            country_coords = {
                'Afghanistan': [33.93911, 67.709953],
                'Albania': [41.153332, 20.168331],
                'Algeria': [28.033886, 1.659626],
                'Andorra': [42.546245, 1.601554],
                'Angola': [-11.202692, 17.873887],
                'Antigua and Barbuda': [17.060816, -61.796428],
                'Argentina': [-38.416097, -63.616672],
                'Armenia': [40.069099, 45.038189],
                'Australia': [-25.274398, 133.775136],
                'Austria': [47.516231, 14.550072],
                'Azerbaijan': [40.143105, 47.576927],
                'Bahamas': [25.03428, -77.39628],
                'Bahrain': [25.930414, 50.637772],
                'Bangladesh': [23.685, 90.3563],
                'Barbados': [13.193887, -59.543198],
                'Belarus': [53.709807, 27.953389],
                'Belgium': [50.503887, 4.469936],
                'Belize': [17.189877, -88.49765],
                'Benin': [9.30769, 2.315834],
                'Bhutan': [27.514162, 90.433601],
                'Bolivia': [-16.290154, -63.588653],
                'Bosnia and Herzegovina': [43.915886, 17.679076],
                'Botswana': [-22.328474, 24.684866],
                'Brazil': [-14.235004, -51.92528],
                'Brunei': [4.535277, 114.727669],
                'Bulgaria': [42.733883, 25.48583],
                'Burkina Faso': [12.238333, -1.561593],
                'Burundi': [-3.373056, 29.918886],
                'Cabo Verde': [16.002082, -24.013197],
                'Cambodia': [12.565679, 104.990963],
                'Cameroon': [7.369722, 12.354722],
                'Canada': [56.130366, -106.346771],
                'Central African Republic': [6.611111, 20.939444],
                'Chad': [15.454166, 18.732207],
                'Chile': [-35.675147, -71.542969],
                'China': [35.86166, 104.195397],
                'Colombia': [4.570868, -74.297333],
                'Comoros': [-11.6455, 43.3333],
                'Congo (Congo-Brazzaville)': [-0.228021, 15.827659],
                'Costa Rica': [9.748917, -83.753428],
                'Croatia': [45.1, 15.2],
                'Cuba': [21.521757, -77.781167],
                'Cyprus': [35.126413, 33.429859],
                'Czechia': [49.817492, 15.472962],
                'Democratic Republic of the Congo': [-4.038333, 21.758664],
                'Denmark': [56.26392, 9.501785],
                'Djibouti': [11.825138, 42.590275],
                'Dominica': [15.414999, -61.370976],
                'Dominican Republic': [18.735693, -70.162651],
                'Ecuador': [-1.831239, -78.183406],
                'Egypt': [26.820553, 30.802498],
                'El Salvador': [13.794185, -88.89653],
                'Equatorial Guinea': [1.650801, 10.267895],
                'Eritrea': [15.179384, 39.782334],
                'Estonia': [58.595272, 25.013607],
                'Eswatini': [-26.522503, 31.465866],
                'Ethiopia': [9.145, 40.489673],
                'Fiji': [-16.578193, 179.414413],
                'Finland': [61.92411, 25.748151],
                'France': [46.227638, 2.213749],
                'Gabon': [-0.803689, 11.609444],
                'Gambia': [13.443182, -15.310139],
                'Georgia': [42.315407, 43.356892],
                'Germany': [51.165691, 10.451526],
                'Ghana': [7.946527, -1.023194],
                'Greece': [39.074208, 21.824312],
                'Grenada': [12.262776, -61.604171],
                'Guatemala': [15.783471, -90.230759],
                'Guinea': [9.945587, -9.696645],
                'Guinea-Bissau': [11.803749, -15.180413],
                'Guyana': [4.860416, -58.93018],
                'Haiti': [18.971187, -72.285215],
                'Honduras': [15.199999, -86.241905],
                'Hungary': [47.162494, 19.503304],
                'Iceland': [64.963051, -19.020835],
                'India': [20.593684, 78.96288],
                'Indonesia': [-0.789275, 113.921327],
                'Iran': [32.427908, 53.688046],
                'Iraq': [33.223191, 43.679291],
                'Ireland': [53.41291, -8.24389],
                'Israel': [31.046051, 34.851612],
                'Italy': [41.87194, 12.56738],
                'Jamaica': [18.109581, -77.297508],
                'Japan': [36.204824, 138.252924],
                'Jordan': [30.585164, 36.238414],
                'Kazakhstan': [48.019573, 66.923684],
                'Kenya': [-1.292066, 36.821946],
                'Kiribati': [-3.370417, -168.734039],
                'Kuwait': [29.31166, 47.481766],
                'Kyrgyzstan': [41.20438, 74.766098],
                'Laos': [19.85627, 102.495496],
                'Latvia': [56.879635, 24.603189],
                'Lebanon': [33.854721, 35.862285],
                'Lesotho': [-29.609988, 28.233608],
                'Liberia': [6.428055, -9.429499],
                'Libya': [26.3351, 17.228331],
                'Liechtenstein': [47.166, 9.555373],
                'Lithuania': [55.169438, 23.881275],
                'Luxembourg': [49.815273, 6.129583],
                'Madagascar': [-18.766947, 46.869107],
                'Malawi': [-13.254308, 34.301525],
                'Malaysia': [4.210484, 101.975766],
                'Maldives': [3.202778, 73.22068],
                'Mali': [17.570692, -3.996166],
                'Malta': [35.937496, 14.375416],
                'Marshall Islands': [7.131474, 171.184478],
                'Mauritania': [21.00789, -10.940835],
                'Mauritius': [-20.348404, 57.552152],
                'Mexico': [23.634501, -102.552784],
                'Micronesia': [7.425554, 150.550812],
                'Moldova': [47.411631, 28.369885],
                'Monaco': [43.738417, 7.424616],
                'Mongolia': [46.862496, 103.846656],
                'Montenegro': [42.708678, 19.37439],
                'Morocco': [31.791702, -7.09262],
                'Mozambique': [-18.665695, 35.529562],
                'Myanmar (Burma)': [21.916221, 95.955974],
                'Namibia': [-22.95764, 18.49041],
                'Nauru': [-0.522778, 166.931503],
                'Nepal': [28.394857, 84.124008],
                'Netherlands': [52.132633, 5.291266],
                'New Zealand': [-40.900557, 174.885971],
                'Nicaragua': [12.865416, -85.207229],
                'Niger': [17.607789, 8.081666],
                'Nigeria': [9.081999, 8.675277],
                'North Korea': [40.339852, 127.510093],
                'North Macedonia': [41.608635, 21.745275],
                'Norway': [60.472024, 8.468946],
                'Oman': [21.512583, 55.923255],
                'Pakistan': [30.375321, 69.345116],
                'Palau': [7.51498, 134.58252],
                'Palestine': [31.947351, 35.227163],
                'Panama': [8.537981, -80.782127],
                'Papua New Guinea': [-6.314993, 143.95555],
                'Paraguay': [-23.442503, -58.443832],
                'Peru': [-9.189967, -75.015152],
                'Philippines': [12.879721, 121.774017],
                'Poland': [51.919438, 19.145136],
                'Portugal': [39.399872, -8.224454],
                'Qatar': [25.354826, 51.183884],
                'Romania': [45.943161, 24.96676],
                'Russia': [61.52401, 105.318756],
                'Rwanda': [-1.940278, 29.873888],
                'Saint Kitts and Nevis': [17.357822, -62.782998],
                'Saint Lucia': [13.909444, -60.978893],
                'Saint Vincent and the Grenadines': [12.984305, -61.287228],
                'Samoa': [-13.759029, -172.104629],
                'San Marino': [43.933333, 12.450001],
                'Sao Tome and Principe': [0.18636, 6.613081],
                'Saudi Arabia': [23.885942, 45.079162],
                'Senegal': [14.497401, -14.452362],
                'Serbia': [44.016521, 21.005859],
                'Seychelles': [-4.679574, 55.491977],
                'Sierra Leone': [8.460555, -11.779889],
                'Singapore': [1.352083, 103.819836],
                'Slovakia': [48.669026, 19.699024],
                'Slovenia': [46.151241, 14.995463],
                'Solomon Islands': [-9.64571, 160.156194],
                'Somalia': [5.152149, 46.199616],
                'South Africa': [-30.559482, 22.937506],
                'South Korea': [35.907757, 127.766922],
                'South Sudan': [6.8769919, 31.3069788],
                'Spain': [40.463667, -3.74922],
                'Sri Lanka': [7.873054, 80.771797],
                'Sudan': [12.862807, 30.217636],
                'Suriname': [3.919305, -56.027783],
                'Sweden': [60.128161, 18.643501],
                'Switzerland': [46.818188, 8.227512],
                'Syria': [34.802075, 38.996815],
                'Taiwan': [23.69781, 120.960515],
                'Tajikistan': [38.861034, 71.276093],
                'Tanzania': [-6.369028, 34.888822],
                'Thailand': [15.870032, 100.992541],
                'Timor-Leste': [-8.874217, 125.727539],
                'Togo': [8.619543, 0.824782],
                'Tonga': [-21.178986, -175.198242],
                'Trinidad and Tobago': [10.691803, -61.222503],
                'Tunisia': [33.886917, 9.537499],
                'Turkey': [38.963745, 35.243322],
                'Turkmenistan': [38.969719, 59.556278],
                'Tuvalu': [-7.109535, 177.64933],
                'Uganda': [1.373333, 32.290275],
                'Ukraine': [48.379433, 31.16558],
                'United Arab Emirates': [23.424076, 53.847818],
                'United Kingdom': [55.378051, -3.435973],
                'United States': [37.09024, -95.712891],
                'Uruguay': [-32.522779, -55.765835],
                'Uzbekistan': [41.377491, 64.585262],
                'Vanuatu': [-15.376706, 166.959158],
                'Vatican City': [41.902916, 12.453389],
                'Venezuela': [6.42375, -66.58973],
                'Vietnam': [14.058324, 108.277199],
                'Yemen': [15.552727, 48.516388],
                'Zambia': [-13.133897, 27.849332],
                'Zimbabwe': [-19.015438, 29.154857]
            }


            # count frequency of each country
            country_counts = results_df['Country'].value_counts().to_dict()

            # map lat/lon safely
            results_df['lat'] = results_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[0])
            results_df['lon'] = results_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[1])
            results_df['count'] = results_df['Country'].map(lambda x: country_counts.get(x, 1))

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=20,
                    longitude=0,
                    zoom=1,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        'HeatmapLayer',
                        data=results_df,
                        get_position='[lon, lat]',
                        get_weight='count',
                        radiusPixels=60,
                    ),
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=results_df,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius='count * 20000',
                        pickable=True,
                    ),
                ],
                tooltip={"text": "Country: {Country}\nProduct: {Product_Name}\nEntries: {count}".format(Country='{Country}', Product_Name='{Product_Name}', count='{count}')}
            ))

        else:
            st.info("No data to display yet. Score a product first!")
    else:
        st.info("No data to display yet. Score a product first!")


