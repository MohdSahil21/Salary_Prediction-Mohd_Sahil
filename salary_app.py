import streamlit as st
import joblib

MODEL_PATHS = {
    'linear': 'linear_regression_model.pkl',
    'random_forest': 'random_forest_regression_model.pkl',
    'job_title_map': 'job_title_mean_salary_map.pkl',
    'scaler_age': 'scaler_age.pkl',
    'scaler_experience': 'scaler_years_experience.pkl',
    'scaler_job': 'scaler_job_title_mean_encoded.pkl'
}

# Model accuracies
MODEL_ACCURACY = {
    "Linear Regression": 80,
    "Random Forest Regression": 98
}

# Education mapping
EDUCATION_MAP = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}

# Job titles list
JOB_TITLES = [
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Back end Developer',
    'Business Analyst', 'Business Development Manager', 'Business Intelligence Analyst',
    'CEO', 'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager',
    'Copywriter', 'Creative Director', 'Customer Service Manager', 'Customer Service Rep',
    'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Delivery Driver',
    'Digital Marketing Manager', 'Director', 'Director of Engineering', 'Director of Finance',
    'Director of HR', 'Director of Marketing', 'Director of Operations', 'Event Coordinator',
    'Financial Advisor', 'Financial Analyst', 'Financial Manager', 'Front End Developer',
    'Full Stack Engineer', 'Graphic Designer', 'HR Generalist', 'HR Manager',
    'IT Manager', 'IT Support', 'Junior Data Analyst', 'Junior Software Engineer',
    'Marketing Analyst', 'Marketing Director', 'Marketing Manager', 'Network Engineer',
    'Office Manager', 'Operations Manager', 'Product Manager', 'Project Manager',
    'Receptionist', 'Recruiter', 'Research Scientist', 'Sales Manager', 'Sales Representative',
    'Senior Data Scientist', 'Senior Software Engineer', 'Software Developer',
    'Software Engineer', 'UX Designer', 'VP of Finance', 'VP of Operations', 'Web Developer'
]


@st.cache_resource
def load_models():
    """Load all models and preprocessing components with caching"""
    try:
        models = {
            'linear': joblib.load(MODEL_PATHS['linear']),
            'random_forest': joblib.load(MODEL_PATHS['random_forest']),
            'job_title_map': joblib.load(MODEL_PATHS['job_title_map']),
            'scaler_age': joblib.load(MODEL_PATHS['scaler_age']),
            'scaler_experience': joblib.load(MODEL_PATHS['scaler_experience']),
            'scaler_job': joblib.load(MODEL_PATHS['scaler_job'])
        }
        return models
    except Exception as e:
        st.error(f"⚠️ ERROR LOADING MODELS: {str(e)}")
        return None


def preprocess_inputs(models, age, education, experience, job_title, gender):
    """Preprocess user inputs for model prediction"""
    
    # Education encoding
    education_encoded = EDUCATION_MAP.get(education, 0)
    
    # Job title encoding - FIX: Handle dictionary/array properly
    job_title_map = models['job_title_map']
    
    # Check if it's a dictionary or needs conversion
    if hasattr(job_title_map, 'item'):  # numpy array
        # If it's saved as numpy array, convert to dict
        job_mean = float(job_title_map.mean())
    elif isinstance(job_title_map, dict):
        # It's a dictionary
        if job_title in job_title_map:
            job_mean = job_title_map[job_title]
        else:
            # Calculate mean from dictionary values
            job_mean = sum(job_title_map.values()) / len(job_title_map)
    else:
        # Fallback - use a default value
        job_mean = 50000.0
    
    # Scale the job title mean encoding
    job_scaled = models['scaler_job'].transform([[job_mean]])[0][0]
    
    # Gender one-hot encoding
    gender_features = {
        'Female': [1, 0, 0],
        'Male': [0, 1, 0],
        'Other': [0, 0, 1]
    }
    gender_encoded = gender_features.get(gender, [0, 0, 0])
    
    # Scale numerical features
    age_scaled = models['scaler_age'].transform([[age]])[0][0]
    exp_scaled = models['scaler_experience'].transform([[experience]])[0][0]
    
    # Create feature array
    features = [
        [age_scaled, education_encoded, exp_scaled, job_scaled] + gender_encoded
    ]
    
    return features



def main():
    # Header with retro ASCII art
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <pre style='color: #00ff41; font-size: 0.6rem; line-height: 1;'>
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1>💰 SALARY PREDICTOR 💰</h1>", unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.error("⚠️ SYSTEM ERROR: FAILED TO LOAD MODELS")
        return
    
    # ========================================================================
    # SIDEBAR - Model Selection
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## 🎮 SELECT MODEL")
        
        selected_model = st.selectbox(
            "ALGORITHM",
            list(MODEL_ACCURACY.keys()),
            help="Choose your prediction algorithm"
        )
        
        # Display accuracy
        accuracy = MODEL_ACCURACY[selected_model]
        st.markdown("### 📊 PERFORMANCE")
        st.metric(
            label="ACCURACY",
            value=f"{accuracy}%",
            delta="OPTIMAL" if accuracy > 90 else "GOOD"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ INFO")
        st.info(
            "SYSTEM: ML-PREDICT-v1.0\n\n"
            "Uses machine learning algorithms to calculate employee salary "
            "predictions based on input parameters."
        )
    
    # ========================================================================
    # MAIN CONTENT - Input Form
    # ========================================================================
    
    st.markdown("## 📝 *INPUT PARAMETERS*")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "👤 AGE",
            min_value=18,
            max_value=65,
            value=30,
            help="Employee age in years"
        )
        
        education = st.selectbox(
            "🎓 EDUCATION",
            list(EDUCATION_MAP.keys()),
            help="Highest qualification achieved"
        )
        
        experience = st.slider(
            "💼 EXPERIENCE (YEARS)",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5,
            help="Total professional experience"
        )
    
    with col2:
        job_title = st.selectbox(
            "💻 JOB TITLE",
            JOB_TITLES,
            help="Current position"
        )
        
        gender = st.selectbox(
            "⚧ GENDER",
            ["Male", "Female", "Other"],
            help="Gender identity"
        )
    
    # ========================================================================
    # PREDICTION SECTION
    # ========================================================================
    
    st.markdown("---")
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 EXECUTE PREDICTION", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🔮 CALCULATING..."):
            try:
                # Preprocess inputs
                features = preprocess_inputs(
                    models, age, education, experience, job_title, gender
                )
                
                # Make prediction
                if selected_model == "Linear Regression":
                    prediction = models['linear'].predict(features)[0]
                else:
                    prediction = models['random_forest'].predict(features)[0]
                
                # Display result with custom styling
                st.markdown(
                    f"""
                    <div class="result-box">
                        <h2>💵 PREDICTED SALARY</h2>
                        <h2>₹{prediction:,.2f}/month</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Additional insights
                st.success(f"✅ PREDICTION COMPLETE | MODEL: {selected_model.upper()}")
            
            except Exception as e:
                st.error(f"⚠️ PREDICTION ERROR: {str(e)}")

if __name__ == "__main__":
    main()
