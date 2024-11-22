# import libraries
import pandas_datareader as pdr  
import seaborn as sns 
import statsmodels.multivariate.pca as pca  
import streamlit as st  

# Configure additional Matplotlib parameters for consistency
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # Set font style to Roboto
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable minor y-ticks
p["xtick.minor.visible"] = True  # Enable minor x-ticks
p["axes.grid"] = True  # Enable grid lines
p["grid.color"] = "0.5"  # Set grid color
p["grid.linewidth"] = 0.5  # Set grid line width

# Fetch data from FRED for various Treasury rates
df = pdr.data.DataReader(['DGS6MO', 'DGS1',
                          'DGS2', 'DGS5',
                          'DGS7', 'DGS10',
                          'DGS20', 'DGS30'], 
                          data_source='fred',  # Data source: Federal Reserve Economic Data (FRED)
                          start='01-01-2022',  # Start date for data
                          end='12-31-2022')  # End date for data

df = df.dropna()  # Drop rows with missing values

# Rename columns for better readability
df = df.rename(columns={'DGS6MO': '0.5 yr', 
                        'DGS1': '1 yr',
                        'DGS2': '2 yr',
                        'DGS5': '5 yr',
                        'DGS7': '7 yr',
                        'DGS10': '10 yr',
                        'DGS20': '20 yr',
                        'DGS30': '30 yr'})

X_df = df.pct_change()  # Calculate daily percentage changes
X_df = X_df.dropna()  # Remove rows with missing values after calculation

# Sidebar setup in Streamlit for user input
with st.sidebar:
    st.title('Principal Component Analysis')  # Add a title to the sidebar
    num_of_PCs = st.slider('Number of PCs',  # Slider to select the number of principal components
                           min_value=1, 
                           max_value=8, 
                           value=2, step=1)  # Default value: 2

# Perform PCA on the data
pca_model = pca.PCA(X_df, standardize=True)  # Apply PCA with standardization
variance_V = pca_model.eigenvals  # Extract eigenvalues to measure variance explained
explained_var_ratio = pca_model.eigenvals / pca_model.eigenvals.sum()  # Calculate the explained variance ratio

X_df_ = pca_model.project(num_of_PCs)  # Project the data onto the selected number of principal components

# Create a Matplotlib subplot for visualizations
fig, axes = plt.subplots(2, 4, figsize=(8, 4))  # Create a 2x4 grid of subplots with a fixed size
axes = axes.flatten()  # Flatten the axes array for easier iteration

# Iterate through columns and plot original, reconstructed, and residual data
for col_idx, ax_idx in zip(list(X_df_.columns), axes):
    sns.lineplot(X_df_[col_idx], ax=ax_idx)  # Plot reconstructed data (PCA projection)
    sns.lineplot(X_df[col_idx], ax=ax_idx)  # Plot original data
    sns.lineplot(X_df[col_idx] - X_df_[col_idx], c='k', ax=ax_idx)  # Plot residual (difference between original and reconstructed)
    ax_idx.set_xticks([])  # Remove x-axis ticks
    ax_idx.set_yticks([])  # Remove y-axis ticks
    ax_idx.axhline(y=0, c='k')  # Add a horizontal line at y=0 for reference

plt.tight_layout()  # Adjust layout to prevent overlap

st.pyplot(fig)  # Render the plot in the Streamlit app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)