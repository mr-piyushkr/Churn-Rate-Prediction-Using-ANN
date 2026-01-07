# =============================================================================
# IMPORTS - All required libraries for the churn prediction app
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =============================================================================
# PAGE CONFIGURATION - Sets up the Streamlit page with title, icon, and layout
# =============================================================================
st.set_page_config(
    page_title="Churn Prediction | ANN",
    page_icon="📉",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS STYLING - Professional look with modern design elements
# =============================================================================
st.markdown("""
<style>
/* Main background and layout */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Title styling */
.main-title {
    font-size: 48px;
    font-weight: 900;
    color: #1e293b;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.sub-title {
    font-size: 20px;
    color: #64748b;
    text-align: center;
    margin-bottom: 30px;
}

/* Section containers */
.section {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
}

/* Prediction result styling - Dark theme */
.prediction-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fecaca;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    border: 2px solid #dc2626;
}

.prediction-low {
    background: linear-gradient(135deg, #14532d, #166534);
    color: #bbf7d0;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    border: 2px solid #16a34a;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e293b, #334155);
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING - Cached function to load dataset efficiently
# =============================================================================
@st.cache_data
def load_data():
    """Load the churn modeling dataset with caching for performance"""
    return pd.read_csv("Churn_Modelling.csv")

df = load_data()

# =============================================================================
# MODEL TRAINING - Cached ANN model with complete preprocessing pipeline
# =============================================================================
@st.cache_resource
def train_model():
    """
    Train the ANN model with preprocessing pipeline:
    1. Data cleaning (remove irrelevant columns)
    2. Feature encoding (Gender, Geography)
    3. Feature scaling (StandardScaler)
    4. Handle class imbalance (SMOTE)
    5. Train ANN model
    """
    # Data preprocessing
    df_model = df.copy()
    df_model.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df_model["Gender"] = le.fit_transform(df_model["Gender"])
    df_model = pd.get_dummies(df_model, columns=["Geography"], drop_first=True)

    # Separate features and target
    X = df_model.drop("Exited", axis=1)
    y = df_model["Exited"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_scaled, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sm, y_sm, test_size=0.2, random_state=42
    )

    # Build ANN model
    model = Sequential([
        Dense(16, activation="relu", input_shape=(X_train.shape[1],)),  # Input layer
        Dense(8, activation="relu"),   # Hidden layer
        Dense(1, activation="sigmoid") # Output layer for binary classification
    ])

    # Compile and train model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Make predictions
    yp = model.predict(X_test)
    y_pred = (yp > 0.5).astype(int)

    return model, scaler, le, X_test, y_test, y_pred

model, scaler, label_encoder, X_test, y_test, y_pred = train_model()

# =============================================================================
# SIDEBAR NAVIGATION - Multi-page navigation system
# =============================================================================
st.sidebar.markdown("## 📌 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose Section",
    ["🏠 Overview", "📊 Data Insights", "🧠 Model Performance", "🔮 Live Prediction"]
)

# Add model info in sidebar
st.sidebar.markdown("## 🤖 Model Info")
st.sidebar.info(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
st.sidebar.info(f"**Dataset Size:** {len(df):,} customers")
st.sidebar.info(f"**Features:** {len(df.columns)-4} variables")

# =============================================================================
# MAIN HEADER - App title and description
# =============================================================================
st.markdown('<div class="main-title">🏦 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Banking Analytics with Neural Networks</div>', unsafe_allow_html=True)
st.markdown('<hr style="border: 2px solid #334155; margin: 20px 0;">', unsafe_allow_html=True)

# =============================================================================
# OVERVIEW PAGE - Business context and key metrics
# =============================================================================
if page == "🏠 Overview":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Business Problem
    Customer churn is a critical challenge in banking, where acquiring new customers costs **5-25x more** than retaining existing ones.
    
    ## 🚀 Our Solution
    This AI-powered system uses an **Artificial Neural Network (ANN)** to predict customer churn with high accuracy, enabling:
    - **Proactive retention strategies**
    - **Targeted marketing campaigns** 
    - **Revenue protection**
    
    ## 🔬 Technical Approach
    - **Deep Learning:** 3-layer neural network
    - **Data Balancing:** SMOTE technique for imbalanced classes
    - **Feature Engineering:** Categorical encoding and scaling
    - **Real-time Predictions:** Interactive customer assessment
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Dataset Size", f"{len(df):,}", "customers")
    with col2:
        st.metric("🎯 Model Accuracy", f"{accuracy_score(y_test, y_pred):.1%}", "on test data")
    with col3:
        st.metric("⚡ Model Type", "ANN", "3 layers")
    with col4:
        churn_rate = (df['Exited'].sum() / len(df)) * 100
        st.metric("📈 Churn Rate", f"{churn_rate:.1f}%", "in dataset")

# =============================================================================
# DATA INSIGHTS PAGE - Visualizations and data exploration
# =============================================================================
elif page == "📊 Data Insights":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📈 Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        churn_counts = df['Exited'].value_counts()
        colors = ['#10b981', '#ef4444']
        ax1.pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig1)
    
    with col2:
        # Age distribution by churn
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x='Exited', y='Age', palette=['#10b981', '#ef4444'], ax=ax2)
        ax2.set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Churn Status (0=Retained, 1=Churned)')
        st.pyplot(fig2)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional insights
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Geography analysis
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        geo_churn = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
        bars = ax3.bar(geo_churn.index, geo_churn.values, color=['#ef4444', '#f59e0b', '#10b981'])
        ax3.set_title('Churn Rate by Geography', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Churn Rate')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        st.pyplot(fig3)
    
    with col2:
        # Gender analysis
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        gender_churn = df.groupby('Gender')['Exited'].mean()
        bars = ax4.bar(gender_churn.index, gender_churn.values, color=['#3b82f6', '#ec4899'])
        ax4.set_title('Churn Rate by Gender', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Churn Rate')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        st.pyplot(fig4)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MODEL PERFORMANCE PAGE - Detailed model evaluation metrics
# =============================================================================
elif page == "🧠 Model Performance":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🎯 Model Performance Metrics")
    
    # Performance metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("🎯 Accuracy", f"{accuracy:.3f}", f"{accuracy:.1%}")
    
    with col2:
        from sklearn.metrics import precision_score
        precision = precision_score(y_test, y_pred)
        st.metric("🔍 Precision", f"{precision:.3f}", f"{precision:.1%}")
    
    with col3:
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, y_pred)
        st.metric("📊 Recall", f"{recall:.3f}", f"{recall:.1%}")
    
    with col4:
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)
        st.metric("⚖️ F1-Score", f"{f1:.3f}", f"{f1:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("🔥 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlBu_r", ax=ax,
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'])
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# LIVE PREDICTION PAGE - Interactive customer churn prediction
# =============================================================================
elif page == "🔮 Live Prediction":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🔮 Customer Churn Prediction")
    st.markdown("Enter customer details below to predict churn probability:")
    
    # Input form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider("💳 Credit Score", 300, 900, 650, help="Customer's credit score (300-900)")
        age = st.slider("👤 Age", 18, 100, 40, help="Customer's age in years")
        tenure = st.slider("⏰ Tenure (Years)", 0, 10, 5, help="Years as bank customer")
        num_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], index=1, help="Number of bank products used")
        has_credit_card = st.selectbox("💳 Has Credit Card", ["No", "Yes"], index=1)
    
    with col2:
        geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"], help="Customer's country")
        gender = st.selectbox("👥 Gender", ["Male", "Female"], help="Customer's gender")
        balance = st.number_input("💰 Account Balance ($)", 0.0, 300000.0, 50000.0, help="Current account balance")
        salary = st.number_input("💼 Estimated Salary ($)", 0.0, 200000.0, 75000.0, help="Annual estimated salary")
        is_active = st.selectbox("🔄 Is Active Member", ["No", "Yes"], index=1, help="Active bank member status")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button and results
    if st.button("🚀 Predict Churn Risk", type="primary"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [1 if gender == 'Male' else 0],  # Encoded like in training
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
            'IsActiveMember': [1 if is_active == 'Yes' else 0],
            'EstimatedSalary': [salary],
            'Geography_Germany': [1 if geography == 'Germany' else 0],
            'Geography_Spain': [1 if geography == 'Spain' else 0]
        })
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        
        # Display results
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎲 Churn Probability", f"{prediction_prob:.1%}")
        
        with col2:
            risk_level = "HIGH" if prediction_prob > 0.7 else "MEDIUM" if prediction_prob > 0.3 else "LOW"
            st.metric("⚠️ Risk Level", risk_level)
        
        with col3:
            st.metric("📊 Confidence", f"{max(prediction_prob, 1-prediction_prob):.1%}")
        
        # Prediction interpretation
        if prediction_class == 1:
            st.markdown(f'<div class="prediction-high">⚠️ HIGH CHURN RISK: This customer has a {prediction_prob:.1%} probability of churning. Immediate retention action recommended!</div>', unsafe_allow_html=True)
            st.markdown("""
            **Recommended Actions:**
            - 📞 Contact customer immediately
            - 🎁 Offer personalized retention incentives
            - 📋 Schedule account review meeting
            - 💰 Consider fee waivers or rate improvements
            """)
        else:
            st.markdown(f'<div class="prediction-low">✅ LOW CHURN RISK: This customer has a {prediction_prob:.1%} probability of churning. Customer appears stable.</div>', unsafe_allow_html=True)
            st.markdown("""
            **Recommended Actions:**
            - 😊 Continue standard customer service
            - 📈 Consider upselling opportunities
            - 📧 Include in regular marketing campaigns
            - 🔄 Monitor for any changes in behavior
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
