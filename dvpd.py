import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io

st.set_page_config(
    page_title="Research Project Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Use a slightly darker background for a modern feel */
    .stApp {
        background-color: #F8F9FA; 
    }
    
    /* Main Streamlit container adjustments */
    .main .block-container {
        padding-top: 1.5rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 0rem;
    }

    /* KPI Cards Styling: White background, subtle border/shadow */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); /* More noticeable shadow */
        border-left: 6px solid #FF4B4B; /* Streamlit Primary Red/Accent */
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        border-left: 6px solid #1E88E5; /* Secondary accent color on hover */
    }
    div[data-testid="stMetric"] > label {
        font-size: 1rem;
        color: #495057;
        font-weight: 500;
    }
    div[data-testid="stMetric"] > div {
        font-size: 1.8rem;
        font-weight: 700;
        color: #343A40;
    }

    /* Adjust sidebar width for better filter visibility */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        background-color: #E9ECEF; /* Light background for sidebar */
    }

    /* Chart Headers */
    h2 {
        color: #343A40;
        border-bottom: 2px solid #FF4B4B; /* Accent color under title */
        padding-bottom: 8px;
        margin-top: 20px;
        font-weight: 600;
    }
    h3 {
        color: #495057;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)



def find_column_by_keyword(df, keyword):
    """Finds the first column whose name contains the specified keyword (case-insensitive)."""
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    return None


@st.cache_data
def load_and_clean_data(uploaded_file):
    """Loads, cleans, and standardizes the research project data from an uploaded file."""
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['csv', 'txt']:
            df = pd.read_csv(uploaded_file, encoding='utf-8', header=0)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel (XLSX/XLS) file.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Department'}, inplace=True)
        
    if df.shape[0] >= 2:
        df = df.iloc[2:].reset_index(drop=True)
        
    if 'Department' in df.columns:
        df['Department'].ffill(inplace=True)


    df.dropna(axis=1, how='all', inplace=True)

    df.columns = df.columns.astype(str).str.strip().str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_', regex=False)
    
    required_keywords = {
        'Department': ['Department'], 
        'Domain': ['Domain', 'Area', 'Field'],
        'Amount_Lakhs': ['Amt', 'Amount', 'Lakhs', 'Funding'],
        'Type': ['Project_Type', 'Type'],
        'Faculty': ['Faculty_Name', 'Faculty', 'PI'],
        'Status': ['Status'],
        'Title': ['Title'],
        'Funding_Agency': ['Funding_Agency', 'Agency', 'Sponsor']
    }
    
    mapped_columns = {}
    
    for target_col, keywords in required_keywords.items():
        for keyword in keywords:
            found_col = find_column_by_keyword(df, keyword)
            if found_col is not None and found_col not in mapped_columns.values():
                mapped_columns[found_col] = target_col
                break

    df.rename(columns=mapped_columns, inplace=True)
    
    essential_cols = ['Department', 'Domain', 'Amount_Lakhs', 'Status', 'Title']
    missing_cols = [col for col in essential_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing essential columns in the uploaded file: {', '.join(missing_cols)}. Please ensure the sheet contains columns for Department, Domain, Funding Amount, Status, and Project Title.")
        return pd.DataFrame()

    
    df.dropna(subset=['Domain'], inplace=True)

    df['Amount_Lakhs'] = pd.to_numeric(df['Amount_Lakhs'], errors='coerce')
    df['Amount_Lakhs'].fillna(0, inplace=True) # Fill missing amounts with 0

    # Clean up categorical columns
    categorical_cols = ['Department', 'Domain', 'Type', 'Funding_Agency', 'Status', 'Faculty']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace('nan', 'Unknown')
            df[col] = df[col].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

    if 'From' in df.columns and 'To' in df.columns:
        df['From'] = pd.to_numeric(df['From'], errors='coerce')
        df['To'] = pd.to_numeric(df['To'], errors='coerce')
        df['Duration_Years'] = df['To'] - df['From']
        df['Duration_Years'].fillna(0, inplace=True)

    return df


st.title("🔬 Research Project Portfolio Analysis")
st.markdown("Interactive Dashboard for Project Metrics")


st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your research data (CSV or XLSX)",
    type=['csv', 'xlsx', 'xls']
)

data = load_and_clean_data(uploaded_file)


if data.empty:
    if uploaded_file is None:
        st.info("Please upload a data file (CSV or XLSX) in the sidebar to visualize the dashboard.")
    else:
        pass 
else:
    st.markdown("### Interactive Dashboard for Project Metrics")

    st.sidebar.header("Filter Options")
    
    department_options = sorted(data['Department'].unique())
    selected_departments = st.sidebar.multiselect(
        "Select Department",
        options=department_options,
        default=department_options
    )

    status_options = sorted(data['Status'].unique())
    selected_statuses = st.sidebar.multiselect(
        "Select Project Status",
        options=status_options,
        default=status_options
    )

    faculty_options = sorted(data['Faculty'].unique())
    selected_faculties = st.sidebar.multiselect(
        "Select Faculty",
        options=faculty_options,
        default=faculty_options
    )

    agency_options = sorted(data['Funding_Agency'].unique())
    selected_agencies = st.sidebar.multiselect(
        "Select Funding Agency",
        options=agency_options,
        default=agency_options
    )

    df_filtered = data[
        data['Department'].isin(selected_departments) & 
        data['Status'].isin(selected_statuses) &
        data['Faculty'].isin(selected_faculties) &
        data['Funding_Agency'].isin(selected_agencies)
    ]

    if df_filtered.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
    else:
        total_projects = len(df_filtered)
        total_funding = df_filtered['Amount_Lakhs'].sum()
        avg_funding = df_filtered['Amount_Lakhs'].mean()
        
        if 'Status' in df_filtered.columns:
            completed_count = df_filtered[df_filtered['Status'] == 'Completed'].shape[0]
        else:
            completed_count = 0


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Projects",
                value=f"{total_projects:,}",
                delta=f"{completed_count} Completed"
            )

        with col2:
            st.metric(
                label="Total Funding Awarded",
                value=f"₹{total_funding:,.2f} Lakhs",
                help="Sum of 'Amount (in lakhs)' for all filtered projects."
            )

        with col3:
            st.metric(
                label="Average Funding Per Project",
                value=f"₹{avg_funding:,.2f} Lakhs"
            )

        with col4:
            if 'Domain' in df_filtered.columns:
                most_common_domain = df_filtered['Domain'].mode().iloc[0] if not df_filtered['Domain'].mode().empty else 'N/A'
            else:
                most_common_domain = 'N/A'
                
            st.metric(
                label="Primary Research Area",
                value=most_common_domain
            )

        st.markdown("---")


        
        dept_chart_col1, dept_chart_col2 = st.columns(2)

        with dept_chart_col1:
            st.subheader("Total Funding by Department (Top Down)")
            
            if 'Department' in df_filtered.columns and 'Amount_Lakhs' in df_filtered.columns:
                dept_funding = df_filtered.groupby('Department')['Amount_Lakhs'].sum().reset_index()
                dept_funding.columns = ['Department', 'Total Funding (Lakhs)']
                
                dept_funding.sort_values(by='Total Funding (Lakhs)', ascending=False, inplace=True)
    
                fig_dept_funding = px.bar(
                    dept_funding,
                    x='Total Funding (Lakhs)', 
                    y='Department', 
                    color='Total Funding (Lakhs)',
                    orientation='h',
                    color_continuous_scale=px.colors.sequential.Sunset, 
                    title='Total Research Funding by Department',
                    labels={'Department': 'Department', 'Total Funding (Lakhs)': 'Total Funding (₹ Lakhs)'},
                    height=450
                )
                
                fig_dept_funding.update_traces(
                    hovertemplate='<b>%{y}</b><br>Funding: ₹%{x:.2f} Lakhs<extra></extra>',
                    marker_line_width=1, 
                    marker_line_color='rgb(10,10,10)', 
                    opacity=0.9
                )
    
                fig_dept_funding.update_layout(
                    xaxis_title='Total Funding (₹ Lakhs)',
                    yaxis_title='Department',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#343A40"),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                fig_dept_funding.update_yaxes(autorange='reversed')
    
                st.plotly_chart(fig_dept_funding, use_container_width=True)
            else:
                st.warning("Cannot display Department Funding chart: 'Department' or 'Amount_Lakhs' column is missing or misidentified.")
                
        with dept_chart_col2:
            st.subheader("Total Project Count by Department (Top Down)")

            if 'Department' in df_filtered.columns:
                dept_project_count = df_filtered['Department'].value_counts().reset_index()
                dept_project_count.columns = ['Department', 'Project Count']
                
                dept_project_count.sort_values(by='Project Count', ascending=False, inplace=True)

                fig_dept_count = px.bar(
                    dept_project_count,
                    x='Project Count', 
                    y='Department', 
                    color='Project Count',
                    orientation='h',
                    color_continuous_scale=px.colors.sequential.Viridis, 
                    title='Total Research Projects by Department',
                    labels={'Department': 'Department', 'Project Count': 'Number of Projects'},
                    height=450
                )
                
                fig_dept_count.update_traces(
                    hovertemplate='<b>%{y}</b><br>Projects: %{x} <extra></extra>',
                    marker_line_width=1, 
                    marker_line_color='rgb(10,10,10)', 
                    opacity=0.9
                )

                fig_dept_count.update_layout(
                    xaxis_title='Number of Projects',
                    yaxis_title='Department', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#343A40"),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                fig_dept_count.update_yaxes(autorange='reversed')

                st.plotly_chart(fig_dept_count, use_container_width=True)
            else:
                st.warning("Cannot display Department Project Count chart: 'Department' column is missing or misidentified.")

        st.markdown("---") 
        
       
        st.subheader("Research Domain Breakdown for Selected Filters")

        if 'Domain' in df_filtered.columns:
            
            domain_breakdown = df_filtered['Domain'].value_counts().reset_index()
            domain_breakdown.columns = ['Domain', 'Project Count']
            
            
            domain_breakdown.sort_values(by='Project Count', ascending=True, inplace=True)

           
            fig_domain_breakdown = px.bar(
                domain_breakdown,
                x='Project Count', 
                y='Domain', 
                color='Project Count',
                orientation='h',
                color_continuous_scale=px.colors.sequential.Plotly3, 
                title='Project Count by Domain (Filtered View)',
                labels={'Domain': 'Research Domain', 'Project Count': 'Number of Projects'},
                height=450 # Set a fixed height
            )
            
            fig_domain_breakdown.update_traces(
                hovertemplate='<b>%{y}</b><br>Projects: %{x} <extra></extra>',
                marker_line_width=1, 
                marker_line_color='rgb(10,10,10)', 
                opacity=0.9
            )

            fig_domain_breakdown.update_layout(
                xaxis_title='Number of Projects',
                yaxis_title='Research Domain', 
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#343A40"),
                margin=dict(t=50, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig_domain_breakdown, use_container_width=True)
        else:
            st.warning("Cannot display Domain Breakdown chart: 'Domain' column is missing or misidentified.")
            
        st.markdown("---")
        
        
        chart_col1, chart_col2 = st.columns([7, 4])


        with chart_col1:
            st.subheader("Project Status Breakdown")

            if 'Status' in df_filtered.columns:
                status_data = df_filtered['Status'].value_counts().reset_index()
                status_data.columns = ['Status', 'Count']
    
                fig_pie = px.pie(
                    status_data,
                    values='Count',
                    names='Status',
                    title='Status Distribution',
                    color_discrete_sequence=px.colors.qualitative.Bold, 
                    hole=0.4 
                )
    
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
                )
                fig_pie.update_layout(
                    showlegend=True,
                    margin=dict(t=50, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#343A40")
                )
    
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Cannot display Status chart: 'Status' column is missing or misidentified.")


        with chart_col2:
            st.markdown(" ") 


        st.markdown("---")

        st.subheader("Filtered Project Details")
        
        display_cols_map = {
            'Title': 'Title',
            'Department': 'Department', 
            'Domain': 'Domain',
            'Type': 'Type',
            'Amount_Lakhs': 'Amount_Lakhs',
            'Status': 'Status',
            'Faculty': 'Faculty',
            'Funding_Agency': 'Funding_Agency'
        }
        
        display_cols = [col for col in display_cols_map.keys() if col in df_filtered.columns]

        if display_cols:
            st.dataframe(
                df_filtered[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No relevant columns found in the dataset to display details.")