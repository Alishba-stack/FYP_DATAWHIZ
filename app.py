
from flask import Flask, jsonify, redirect, render_template,session, request, send_file, url_for
from flask_sqlalchemy import SQLAlchemy
import openai
import re
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tkinter import messagebox
import os, io
import pdfkit
import pandas as pd
import seaborn as sns
import random
import numpy as np
from scipy.stats import gaussian_kde
import json
from fpdf import FPDF
from scipy import stats
from scipy.stats import ttest_ind
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ks_2samp
from scipy.stats import f_oneway
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
from matplotlib.sankey import Sankey
from scipy.stats import mannwhitneyu
import networkx as nx
import squarify
from scipy.stats import norm
import statsmodels.api as sm
from pandas.plotting import andrews_curves
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from flask_migrate import Migrate
import base64
from io import BytesIO
from sqlalchemy.exc import IntegrityError
from flask import flash
import openpyxl

# Create a Flask app instance
app = Flask(__name__, static_url_path='/statics')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Function to validate username
def is_valid_username(username):
    errors = []
    if len(username) < 6 or len(username) > 20:
        errors.append("Username must be between 6 and 20 characters.")
    if ' ' in username:
        errors.append("Username cannot contain spaces.")
    if not re.match(r'^[A-Za-z0-9@#$%^&+=!]*$', username):
        errors.append("Username can only contain letters, numbers, and symbols (@#$%^&+=!).")
    if not re.search(r'[A-Z]', username):
        errors.append("Username must contain at least one uppercase letter (A-Z).")
    if not re.search(r'[a-z]', username):
        errors.append("Username must contain at least one lowercase letter (a-z).")
    if not re.search(r'[0-9]', username):
        errors.append("Username must contain at least one digit (0-9).")
    return errors

# Function to validate password
def is_valid_password(password):
    errors = []
    if len(password) < 6 or len(password) > 20:
        errors.append("Password must be between 6 and 20 characters.")
    if not re.match(r'^[A-Za-z0-9~`!@#$%^&*()+=_\-{}[\]|:;”’?/<>,.]*$', password):
        errors.append("Password can only contain letters, numbers, and special characters (~`!@#$%^&*()+=_-{}[]\\|:;”’?/<>,.).")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter (A-Z).")
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter (a-z).")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one digit (0-9).")
    if not re.search(r'[~`!@#$%^&*()+=_\-{}[\]|:;”’?/<>,.]', password):
        errors.append("Password must contain at least one special character (~`!@#$%^&*()+=_-{}[]\\|:;”’?/<>,.).")
    return errors

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username
            return redirect(url_for('solutions'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirmPassword']
        
        errors = []
        
        username_errors = is_valid_username(username)
        if username_errors:
            errors.extend(username_errors)
        
        password_errors = is_valid_password(password)
        if password_errors:
            errors.extend(password_errors)
        
        if password != confirm_password:
            errors.append("Passwords do not match.")
        
        if User.query.filter_by(username=username).first():
            errors.append("Username already exists. Please choose a different one.")
        
        if errors:
            return render_template('signup.html', error_messages=errors)
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        session['username'] = username
        return redirect(url_for('solutions'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    # Clear session data
    session.clear()
    # Redirect to the login page after logout
    return redirect(url_for('login'))

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if request.method == 'POST':
        username = session['username']
        old_password = request.form['old_password']
        new_password = request.form['new_password']
        
        # Check if the old password provided matches the user's current password
        user = User.query.filter_by(username=username).first()
        if user and user.password == old_password:
            # Update the user's password with the new password
            user.password = new_password
            db.session.commit()
            return redirect(url_for('solutions'))
    
    return render_template('change_password.html')


@app.route('/users')
def view_users():
    # Query all users from the database
    users = User.query.all()
    return render_template('users.html', users=users)

# Set the secret key for the application
app.secret_key = 'datawhiz'

# Set the uploads directory and allowed file extensions
UPLOADS_DIR = 'uploads'  # Set your actual uploads directory
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
        openai.api_key = self.api_key

    def generate_text(self, prompt, max_tokens=50, temperature=0.7):
        if prompt is None or prompt.strip() == "":
            return "Prompt is empty or None"
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return str(e)

    @staticmethod
    def read_dataset(file_path):
        file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))
        if file_path.endswith('.csv'):
            dataset = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            dataset = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        return dataset

class AskWhizVisualization:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        label_encoders = {}
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
        return df, label_encoders
    
    @staticmethod
    def generate_chart(dataset_path, chart_type):
        # Load the dataset
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
        original_df = df.copy() 
        
        # Label encode non-numeric columns and get mappings
        df, label_encoders = AskWhizVisualization.label_encode_non_numeric_columns(df)

        # Configure Matplotlib to use a non-GUI backend
        plt.switch_backend('agg')
        plt.ioff()

        # Generate the corresponding chart based on the selected chart type
        if chart_type == 'histogram':
            # Generate histograms for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 8))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                ax.hist(df[column], bins=20, edgecolor='k')

                if column in label_encoders:
                    # Use original labels for categorical data
                    le = label_encoders[column]
                    ax.set_xticks(range(len(le.classes_)))
                    ax.set_xticklabels(le.classes_, rotation=90)

                ax.set_xlabel(original_df[column].name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'box_plot':
            # Generate box plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 8))
            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                df[[column]].boxplot(ax=ax, vert=False, grid=False)
                
                if column in label_encoders:
                    # Use original labels for categorical data
                    le = label_encoders[column]
                    ax.set_yticks(range(len(le.classes_)))
                    ax.set_yticklabels(le.classes_)

                ax.set_xlabel('Value')
                ax.set_ylabel(original_df[column].name)
                ax.set_title(f'Box Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'violin_plot':
            # Generate violin plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 8))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                sns.violinplot(data=df[[column]], ax=ax)
                
                if column in label_encoders:
                    # Use original labels for categorical data
                    le = label_encoders[column]
                    ax.set_xticks(range(len(le.classes_)))
                    ax.set_xticklabels(le.classes_, rotation=45)

                ax.set_xlabel(original_df[column].name)
                ax.set_ylabel('Value')
                ax.set_title(f'Violin Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'dot_plot':
            # Generate dot plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 8))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                sns.stripplot(data=df[[column]], ax=ax, jitter=True)
                
                if column in label_encoders:
                    # Use original labels for categorical data
                    le = label_encoders[column]
                    ax.set_xticks(range(len(le.classes_)))
                    ax.set_xticklabels(le.classes_, rotation=45)

                ax.set_xlabel(original_df[column].name)
                ax.set_ylabel('Value')
                ax.set_title(f'Dot Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'quantile-quantile_plot':
            # Generate Q-Q plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                sm.qqplot(df[column], line ='45', ax=ax)

                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')
                ax.set_title(f'Q-Q Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'density_plot':
            # Generate density plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                sns.kdeplot(data=df[column], ax=ax)

                ax.set_xlabel(original_df[column].name)
                ax.set_ylabel('Density')
                ax.set_title(f'Density Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'distribution_plot':
            # Generate distribution plots for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]
                sns.histplot(df[column], kde=True, ax=ax, stat='density')

                ax.set_xlabel(original_df[column].name)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Plot for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'scatter_plot':
            # Generate scatter plot for each pair of numeric columns
            numeric_columns = df.select_dtypes(include=np.number).columns
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i != j:
                        plt.scatter(df[col1], df[col2], alpha=0.5, label=f'{col1} vs {col2}')

            plt.title('Scatter Plot')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
        elif chart_type == 'line_plot':
            df.plot()
            plt.title('Line Plot for All Columns')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend().set_visible(False)
        elif chart_type == 'bar_chart':
            # Generate the histogram for the entire dataset
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column].plot(kind='hist', alpha=0.5, label=column)

            plt.title('Bar Chart')
            plt.xlabel('Values')
        elif chart_type == 'pie_chart':
            # Generate pie charts for all columns
            num_cols = len(df.columns)
            num_rows = (num_cols + 1) // 2  # Adjust the number of rows for an even number of columns
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

            for i, column in enumerate(df.columns):
                ax = axs[i // 2, i % 2] if num_rows > 1 else axs[i % 2]

                if column in label_encoders:
                    # Handle categorical columns
                    data = df[column][df[column] >= 0]  # Skip negative values
                    counts = data.value_counts()
                    le = label_encoders[column]
                    labels = le.inverse_transform(counts.index)
                else:
                    # Handle numeric columns with bins
                    data = df[column][df[column] >= 0]  # Skip negative values
                    bins = pd.cut(data, bins=5)
                    counts = bins.value_counts()
                    labels = counts.index.astype(str)

                # Generate pie chart
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
                ax.set_title(f'Pie Chart for {original_df[column].name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'correlation_matrix':
            # Compute correlation matrix
            corr = df.corr()

            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(15, 10))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            # Draw the heatmap with the correct aspect ratio
            sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f')

            # Set original labels for the axes
            labels = []
            for column in df.columns:
                if column in label_encoders:
                    labels.append(original_df[column].name)
                else:
                    labels.append(original_df[column].name)

            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
        elif chart_type == 'heatmap':
            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(15, 10))

            # Generate a custom colormap
            cmap = sns.color_palette("coolwarm", as_cmap=True)

            # Draw the heatmap with the correct aspect ratio
            sns.heatmap(df.corr(), cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f')

            # Set original labels for the axes
            labels = []
            for column in df.columns:
                if column in label_encoders:
                    labels.append(original_df[column].name)
                else:
                    labels.append(original_df[column].name)

            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

            plt.title('Heatmap of Dataset')
            plt.tight_layout()
            plt.show()
        elif chart_type == '3d_scatter_plot':
            # Set up the figure and 3D axis
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D scatter plot for each combination of columns
            for i in range(len(df.columns)):
                for j in range(i+1, len(df.columns)):
                    for k in range(j+1, len(df.columns)):
                        ax.scatter(df.iloc[:, i], df.iloc[:, j], df.iloc[:, k], c='b', marker='o')

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Scatter Plot of Entire Dataset')

            plt.show()
        elif chart_type == 'parallel_coordinate_plot':
            # Create parallel coordinate plot
            plt.figure(figsize=(10, 6))
            parallel_coordinates(df, df.columns[-1], colormap='viridis')

            # Rotate x-axis labels
            plt.xticks(rotation=90)

            # Turn off grid and legend
            plt.grid(False)
            plt.legend().set_visible(False)

            # Set title
            plt.title('Parallel Coordinate Plot')

            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()
        elif chart_type == 'andrews_plot':
            plt.subplots_adjust(top=0.9)
            plt.figure(figsize=(10, 6))
            andrews_curves(df, df.columns[0], colormap='viridis')
            plt.grid(False)
            plt.legend().set_visible(False)
            plt.title('Andrews Plot')
        elif chart_type == 'radar_chart':
            num_vars = df.shape[1]
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            values = df.values
            values = np.concatenate((values, values[:, [0]]), axis=1)
            angles += angles[:1]
            plt.figure(figsize=(8, 8))
            for i in range(len(values)):
                plt.polar(angles, values[i], marker='o', label=f'{df.index[i]}')
            plt.legend().set_visible(False)
            plt.thetagrids(np.degrees(angles[:-1]), df.columns)
            plt.tight_layout()
            plt.title('Radar Chart')
        elif chart_type == 'chernoff_faces':
            normalized_df = (df - df.min()) / (df.max() - df.min())
            max_faces_to_display = 16
            num_faces = min(len(df), max_faces_to_display)
            faces_per_row = 4
            num_rows = int(np.ceil(num_faces / faces_per_row))
            fig, ax = plt.subplots(num_rows, faces_per_row, figsize=(10, 3 * num_rows))
            for i in range(num_faces):
                row_idx = i // faces_per_row
                col_idx = i % faces_per_row
                params = {
                    'head_radius': normalized_df.iloc[i, 0],
                    'eye_size': normalized_df.iloc[i, 1] if df.shape[1] > 1 else 0.1,
                    'mouth_width': normalized_df.iloc[i, 2] if df.shape[1] > 2 else 0.5,
                    'mouth_smile': normalized_df.iloc[i, 3] if df.shape[1] > 3 else 0.5,
                }
                ax[row_idx, col_idx].set_xlim(-1.5, 1.5)
                ax[row_idx, col_idx].set_ylim(-1.5, 1.5)
                ax[row_idx, col_idx].axis('off')
                circle_head = plt.Circle((0, 0), params['head_radius'], color='yellow', fill=True, ec='black')
                ax[row_idx, col_idx].add_artist(circle_head)
                circle_eye_left = plt.Circle((-0.4, 0.3), params['eye_size'], color='black', fill=True)
                circle_eye_right = plt.Circle((0.4, 0.3), params['eye_size'], color='black', fill=True)
                ax[row_idx, col_idx].add_artist(circle_eye_left)
                ax[row_idx, col_idx].add_artist(circle_eye_right)
                mouth_x = np.linspace(-0.4, 0.4, 100)
                mouth_y = params['mouth_smile'] * np.sin(np.pi * mouth_x / params['mouth_width'])
                ax[row_idx, col_idx].plot(mouth_x, mouth_y, color='black')
            for i in range(num_faces, num_rows * faces_per_row):
                row_idx = i // faces_per_row
                col_idx = i % faces_per_row
                fig.delaxes(ax[row_idx, col_idx])
            plt.suptitle('Chernoff Faces')
            plt.tight_layout()
        elif chart_type == 'tree_map':
            # Sort values to prioritize larger values
            df = df.sort_values(by=df.columns[1], ascending=False)
            
            # Take the top N values to display in the tree map
            N = 200  # Adjust as needed
            df_top = df.head(N)
            
            # Replace 0 values with a small non-zero value to avoid errors
            df_top.iloc[:, 1] = df_top.iloc[:, 1].replace(0, 1)
            
            # Generate the tree map
            plt.figure(figsize=(10, 6))
            squarify.plot(sizes=df_top.iloc[:, 1], label=df_top.iloc[:, 0], alpha=0.8)
            plt.title('Tree Map')
            plt.tight_layout()
            plt.axis('off')
            plt.show()
        elif chart_type == 'sunburst_chart':
            # Sort values to prioritize larger values
            df = df.sort_values(by=df.columns[1], ascending=False)

            # Take the top N values to display in the tree map
            N = 100  # Adjust as needed
            df_top = df.head(N)

            # Replace 0 values with a small non-zero value to avoid errors
            df_top.iloc[:, 1] = df_top.iloc[:, 1].replace(0, 1)

            # Calculate angles and radii for sectors
            values = df_top.iloc[:, 1].values
            labels = df_top.iloc[:, 0].values
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
            radii = np.sqrt(values / np.pi)

            # Define a color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

            # Create subplots and plot the sectors with different colors
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            bars = ax.bar(angles, radii, width=(2*np.pi) / len(values), bottom=0.0, color=colors, edgecolor='black', alpha=0.8)

            # Add labels to each sector with adjusted vertical position
            label_distance = 1.5  # Adjust the distance of labels from the bars
            for bar, angle, label in zip(bars, angles, labels):
                ax.text(angle, bar.get_height() + label_distance, label, ha='center', va='bottom', fontsize=8)

            ax.set_xticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.title('Sunburst Chart')
            plt.show()
        elif chart_type == 'force_directed_graph':
            # Create a figure and adjust its size
            plt.figure(figsize=(10, 8))

            # Create an empty graph
            G = nx.Graph()

            # Add nodes for each column in the DataFrame
            for column in df.columns:
                G.add_node(column)

            # Add edges between each pair of columns
            for column1 in df.columns:
                for column2 in df.columns:
                    if column1 != column2:
                        G.add_edge(column1, column2)

            # Define the layout for the graph (spring layout)
            pos = nx.spring_layout(G)

            # Draw the graph with labels and styling
            nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', edge_color='gray', linewidths=1, font_size=8)

            # Set the title and adjust layout
            plt.title('Force Directed Graph')
            plt.tight_layout()

            # Show the plot
            plt.show()
        elif chart_type == 'sankey_diagram':
            # Define your data
            flows = []  # List to store flows

            # Iterate over columns of the DataFrame
            for i in range(df.shape[1] - 1):
                # Add flows for each pair of columns
                flows.extend([1, -1])

            # Create the Sankey diagram
            fig, ax = plt.subplots(figsize=(10, 8))
            sankey = Sankey(ax=ax, unit=None)
            sankey.add(flows=flows, orientations=[0, 1] * (len(flows) // 2))
            
            sankey.finish()

            # Set title and adjust layout
            plt.title('Sankey Diagram')
            plt.tight_layout()
            plt.show()
        elif chart_type == 'covariance_matrix':
            plt.figure(figsize=(12, 10))
            cov_matrix = df.cov()
            sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Covariance Matrix')
            plt.tight_layout()
            plt.show()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()
   
        return plot_data
    
class AskWhizAnalysis:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            # Convert values to strings before label encoding
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df
    
    @staticmethod
    def eda(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        eda_result = f"Exploratory Data Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Basic information about the dataset
        eda_result += f"Number of rows: {len(df)}\n"
        eda_result += f"Number of columns: {len(df.columns)}\n"
        eda_result += f"All Variables: {', '.join(df.columns)}\n"
        eda_result += "\n"

        # Summary statistics for all numerical columns
        selected_numerical_df = df.select_dtypes(include=['int64', 'float64'])

        if not selected_numerical_df.empty:
            eda_result += "Summary Statistics for All Numerical Columns:\n"
            eda_result += selected_numerical_df.describe().to_string() + "\n"
            eda_result += "\n"

            # Info about data types and missing values for all variables
            eda_result += "Data Types and Missing Values for All Variables:\n"
            selected_info_str = selected_numerical_df.info()
            eda_result += selected_info_str if selected_info_str is not None else "No missing values\n"
            eda_result += "\n"

            # Univariate Analysis - Count of unique values for each column
            eda_result += "Count of Unique Values for Each Column:\n"
            eda_result += selected_numerical_df.nunique().to_string() + "\n"
            eda_result += "\n"

            # Correlation matrix for all numerical columns
            eda_result += "Correlation Matrix for All Numerical Columns:\n"
            eda_result += selected_numerical_df.corr().to_string() + "\n"

        else:
            eda_result += "No numeric columns in the dataset.\n"

        # Data Cleaning and Preprocessing
        eda_result += "Data Cleaning and Preprocessing:\n"
        # Handle missing values
        df_cleaned = df.dropna()
        eda_result += f"Dataset after removing rows with missing values: {len(df_cleaned)} rows\n\n"

        # Display the first few rows of the cleaned dataset
        eda_result += "Preview of the Cleaned Dataset:\n"
        eda_result += df_cleaned.head().to_string() + "\n\n"


        return eda_result

    @staticmethod
    def descriptive_statistics(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        desc_stats_result = f"Descriptive Statistics for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Summary statistics for the entire dataset
        desc_stats_result += df.describe().to_string()

        # Skewness and Kurtosis for each numerical column
        desc_stats_result += "\nSkewness:\n"
        desc_stats_result += df.skew().to_string()

        desc_stats_result += "\n\nKurtosis:\n"
        desc_stats_result += df.kurtosis().to_string()

        # Counts and percentages for categorical columns
        desc_stats_result += "\n\nCategorical Column Statistics:\n"
        for column in df.select_dtypes(include='object'):
            desc_stats_result += f"\n{column}:\n"
            desc_stats_result += df[column].value_counts().to_string() + "\n"

        # Frequency distribution for numerical columns (histograms)
        desc_stats_result += "\n\nNumerical Column Frequency Distributions:\n"
        for column in df.select_dtypes(include=['int64', 'float64']):
            desc_stats_result += f"\n{column}:\n"
            desc_stats_result += np.histogram(df[column], bins='auto')[0].tolist().__str__() + "\n"

        return desc_stats_result

    @staticmethod
    def inferential_statistics(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        inferential_stats_result = f"Inferential Statistics for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Calculate confidence intervals for all numeric columns
        for column in df.select_dtypes(include=['number']).columns:
            # Assuming the variable is numerical
            mean = np.mean(df[column])
            std_dev = np.std(df[column], ddof=1)  # ddof=1 for sample standard deviation
            sample_size = len(df[column])

            confidence_level = 0.95
            margin_of_error = norm.ppf((1 + confidence_level) / 2) * (std_dev / np.sqrt(sample_size))
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error

            inferential_stats_result += f"\nConfidence Interval for the Mean of {column}:\n"
            inferential_stats_result += f"({lower_bound}, {upper_bound})\n"

            # Additional information about the normal distribution
            inferential_stats_result += f"Variable {column} follows a normal distribution.\n"
            inferential_stats_result += f"Probability Density Function (PDF) for {column}:\n"
            x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
            pdf = norm.pdf(x, mean, std_dev)
            inferential_stats_result += f"PDF: {pdf}\n"

        return inferential_stats_result

    @staticmethod
    def hypothesis_testing(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        hypothesis_testing_result = f"Hypothesis Testing for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        numeric_columns = df.select_dtypes(include=['number']).columns

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                variable1_name = numeric_columns[i]
                variable2_name = numeric_columns[j]

                # Extract numerical values for the hypothesis test
                variable1 = df[variable1_name].dropna().values
                variable2 = df[variable2_name].dropna().values

                t_stat, p_value = ttest_ind(variable1, variable2)
                significance_level = 0.05

                hypothesis_testing_result += f"\nT-Statistic for Equality of Means ({variable1_name}, {variable2_name}): {t_stat}\n"
                hypothesis_testing_result += f"P-Value for Equality of Means: {p_value}\n"

                if p_value < significance_level:
                    hypothesis_testing_result += "Reject the null hypothesis for Equality of Means\n"
                else:
                    hypothesis_testing_result += "Fail to reject the null hypothesis for Equality of Means\n"

        # Analyze dataset based on the findings
        if len(numeric_columns) > 0:
            analysis = "The hypothesis testing results indicate whether there are significant differences in the means of numeric columns in the dataset. A rejection of the null hypothesis suggests that there is evidence to support differences in means, while a failure to reject the null hypothesis indicates no significant differences.\n\n"
        else:
            analysis = "No numeric columns were found in the dataset, limiting the application of hypothesis testing for comparing means. Consider exploring alternative analytical methods or acquiring additional data.\n\n"

        hypothesis_testing_result += "Analysis of Dataset:\n"
        hypothesis_testing_result += analysis

        return hypothesis_testing_result

    
    @staticmethod
    def correlation_analysis(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        correlation_analysis_result = f"Correlation Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Calculate the correlation matrix for all numeric columns
        correlation_matrix = df.select_dtypes(include=['number']).corr()

        correlation_analysis_result += f"\nCorrelation Matrix:\n{correlation_matrix}\n"

        # Identify significant correlations (you can customize the threshold)
        significance_level = 0.05
        significant_correlations = correlation_matrix[(correlation_matrix.abs() > significance_level) & (correlation_matrix < 1)]
        
        correlation_analysis_result += f"\nSignificant Correlations (p-value < {significance_level}):\n{significant_correlations}\n"

        # Analyze dataset based on the findings
        if significant_correlations.empty:
            analysis = "The correlation analysis did not find any significant correlations between numeric columns in the dataset. This suggests that there may be no strong linear relationships among the variables, or the sample size might be insufficient to detect significant correlations.\n\n"
        else:
            analysis = "The correlation analysis identified significant correlations between some numeric columns in the dataset. This indicates potential linear relationships among these variables, which may provide insights for further analysis or modeling.\n\n"

        correlation_analysis_result += "Analysis of Dataset:\n"
        correlation_analysis_result += analysis

        return correlation_analysis_result

    @staticmethod
    def regression_analysis(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Assuming you have a dependent variable 'y' and independent variables 'x1', 'x2', ...
        dependent_variable = df.columns[0]  # Assuming the first column is the dependent variable
        independent_variables = df.columns[1:]  # Assuming the rest of the columns are independent variables

        # Add a constant term to the independent variables (required for statsmodels)
        X = sm.add_constant(df[independent_variables])

        # Fit the multiple regression model
        model = sm.OLS(df[dependent_variable], X).fit()

        # Get the regression results
        regression_results = model.summary()

        # Analyze the regression results
        if model.pvalues[1:].max() < 0.05:
            analysis = "The regression analysis indicates that at least one independent variable has a significant impact on the dependent variable. This suggests that the model may be useful for predicting or explaining the variation in the dependent variable.\n\n"
        else:
            analysis = "The regression analysis does not show any independent variables with a significant impact on the dependent variable. This suggests that the model may not be useful for predicting or explaining the variation in the dependent variable.\n\n"

        regression_analysis_result = f"Regression Analysis for {file_path}:\n"
        regression_analysis_result += str(regression_results) + "\n\n"  # Convert Summary object to string
        regression_analysis_result += "Analysis of Regression Results:\n"
        regression_analysis_result += analysis

        return regression_analysis_result


    @staticmethod
    def time_series_analysis(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        time_series_analysis_result = f"Time Series Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        if 'timestamp' in df.columns:
            # Convert 'timestamp' column to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Set 'timestamp' as the index
            df.set_index('timestamp', inplace=True)

            time_series_analysis_result += f"\nSummary Statistics Over Time:\n"

            # Calculate summary statistics for each numeric column
            summary_statistics = df.select_dtypes(include=['number']).describe()

            time_series_analysis_result += f"{summary_statistics}\n"

            # Additional time series analysis can be added based on specific requirements

        else:
            time_series_analysis_result += "No timestamp column found in the dataset for time series analysis.\n"

        return time_series_analysis_result

    @staticmethod
    def clustering(file_path):
        num_clusters = 10
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        clustering_result = f"Clustering Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Select numeric columns for clustering
        numeric_columns = df.select_dtypes(include=['number']).columns
        data_for_clustering = df[numeric_columns]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)

        # Display the cluster distribution
        cluster_distribution = df['cluster'].value_counts().sort_index()
        clustering_result += f"\nCluster Distribution:\n{cluster_distribution}\n"

        # Analyze the clustering result
        if len(cluster_distribution) > 1:
            clustering_result += "The dataset has been successfully clustered into multiple groups, indicating the presence of distinct patterns or subgroups within the data.\n"
        else:
            clustering_result += "The dataset could not be effectively clustered into multiple groups, suggesting homogeneity or lack of distinguishable patterns within the data.\n"

        return clustering_result

    
    @staticmethod
    def classification(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        classification_result = f"Classification Analysis for {file_path}:\n"

        # Assume the first column as the target variable
        target_column = df.columns[0]

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Select features and target variable
        features = df.drop(columns=[target_column])
        target = df[target_column]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Initialize and train a Random Forest Classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_str = classification_report(y_test, y_pred)

        classification_result += f"\nAccuracy: {accuracy:.2f}\n"
        classification_result += f"\nClassification Report:\n{classification_report_str}\n"

        # Analyze the classification result
        if accuracy > 0.8:
            classification_result += "The classifier achieved high accuracy, indicating that it can effectively distinguish between different classes in the dataset.\n"
            classification_result += "This suggests that the features selected for classification contain significant information for predicting the target variable.\n"
        else:
            classification_result += "The classifier achieved relatively low accuracy, suggesting that it may struggle to accurately classify instances in the dataset.\n"
            classification_result += "This could indicate either a lack of discriminatory power in the features or a need for more sophisticated modeling techniques.\n"

        return classification_result

    @staticmethod
    def dimensionality_reduction(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizAnalysis.label_encode_non_numeric_columns(df)

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Perform PCA
        pca = PCA()  # No need to specify n_components
        X_pca = pca.fit_transform(X_scaled)

        # Extract the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Calculate the cumulative explained variance
        cumulative_explained_variance = explained_variance_ratio.cumsum()

        # Extract the principal component loadings
        principal_component_loadings = pca.components_

        # Generate textual results
        result = f"Dimensionality Reduction Analysis for {file_path}:\n"
        result += f"\nExplained Variance Ratio: {explained_variance_ratio}\n"
        result += f"\nCumulative Explained Variance: {cumulative_explained_variance}\n"
        result += f"\nPrincipal Component Loadings:\n{principal_component_loadings}\n"

        # Analyze the dimensionality reduction result
        if len(explained_variance_ratio) > 1 and cumulative_explained_variance[-1] > 0.8:
            result += "The dimensionality reduction suggests that a significant portion of the variance in the dataset can be explained by a reduced number of principal components.\n"
            result += "This indicates that the original high-dimensional dataset may contain redundant or unnecessary features.\n"
        else:
            result += "The dimensionality reduction did not result in a substantial reduction in the number of dimensions or explained variance.\n"
            result += "This suggests that the dataset may not benefit significantly from dimensionality reduction techniques like PCA.\n"

        return result

class AskWhizModel:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df
    
    @staticmethod
    def linear_regression_model(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variables = [col for col in df.columns if col != dependent_variable]

        # Create and fit the linear regression model
        X = df[independent_variables]
        y = df[dependent_variable]
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions using the model
        y_pred = model.predict(X)

        # Optionally, you can print coefficients and intercept
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)

        # Create a scatter plot with the regression line
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Linear Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def multiple_regression_model(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variables = [col for col in df.columns if col != dependent_variable]
        independent_variable, independent_variable2 = np.random.choice(independent_variables, size=2, replace=False)

        # Create and fit the multiple linear regression models
        X1 = df[independent_variable].values.reshape(-1, 1)
        X2 = df[independent_variable2].values.reshape(-1, 1)
        y = df[dependent_variable].values.reshape(-1, 1)
        
        model1 = LinearRegression()
        model1.fit(X1, y)

        model2 = LinearRegression()
        model2.fit(X2, y)

        # Make predictions using the models
        y_pred1 = model1.predict(X1)
        y_pred2 = model2.predict(X2)

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X1, y, color='black', label=f'{independent_variable} vs {dependent_variable}')
        plt.scatter(X2, y, color='red', label=f'{independent_variable2} vs {dependent_variable}')

        # Plot the regression lines
        plt.plot(X1, y_pred1, color='blue', linewidth=3, label=f'Regression Line for {independent_variable}')
        plt.plot(X2, y_pred2, color='green', linewidth=3, label=f'Regression Line for {independent_variable2}')

        # Customize the plot
        plt.title('Multiple Linear Regression')
        plt.xlabel('Independent Variables')
        plt.ylabel('Dependent Variable')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def logistic_regression_model(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variables = [col for col in df.columns if col != dependent_variable]
        independent_variable = np.random.choice(independent_variables)

        # Extract the relevant columns
        X = df[[independent_variable]]
        y = df[dependent_variable]

        # Create and fit the logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # Create a scatter plot with logistic regression curve using seaborn
        sns.regplot(x=X[independent_variable], y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})

        # Customize the plot
        plt.title('Logistic Regression')
        plt.xlabel('Independent Variables')
        plt.ylabel('Dependent Variable')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def poisson_regression_model(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select independent and dependent variables
        independent_variable = np.random.choice(df.columns)
        dependent_variable = np.random.choice([col for col in df.columns if col != independent_variable])
        
        # Extract the relevant columns
        X = df[independent_variable].astype(float)
        y = df[dependent_variable].astype(float)

        # Add a constant term to the independent variable
        X = sm.add_constant(X)

        # Fit the Poisson regression model
        model = sm.GLM(y, X, family=sm.families.Poisson())
        model_result = model.fit()

        # Display the summary of the model
        print(model_result.summary())

        # Plot the observed and predicted values
        plt.scatter(X[independent_variable], y, color='black', label='Observed')
        plt.scatter(X[independent_variable], model_result.predict(), color='red', label='Predicted')
        plt.title('Poisson Regression')
        plt.xlabel('Independent Variables')
        plt.ylabel('Dependent Variable')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def negative_binomial_regression_model(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select independent and dependent variables
        independent_variable = np.random.choice(df.columns)
        dependent_variable = np.random.choice([col for col in df.columns if col != independent_variable])

        # Extract the relevant columns
        X = df[[independent_variable]]
        y = df[dependent_variable]

        # Add a constant term to the independent variable
        X = sm.add_constant(X)

        # Fit the Negative Binomial regression model
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        model_result = model.fit()

        # Display the summary of the model
        print(model_result.summary())

        # Plot observed vs. predicted counts
        predicted_counts = model_result.predict(X)
        observed_counts = y

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(observed_counts)), observed_counts, color='blue', alpha=0.7, label='Observed')
        plt.bar(range(len(predicted_counts)), predicted_counts, color='red', alpha=0.7, label='Predicted')
        plt.title('Negative Binomial Regression')
        plt.xlabel('Observations')
        plt.ylabel('Counts')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def ridge_regression_model(file_path, alpha=1.0):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variable = np.random.choice([col for col in df.columns if col != dependent_variable])

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize and fit the Ridge regression model
        model = Ridge(alpha=alpha)
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the model
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Ridge Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Optionally, you can print coefficients and intercept
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def lasso_regression_model(file_path, alpha=1.0):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variable = np.random.choice([col for col in df.columns if col != dependent_variable])

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize the Lasso regression model
        model = Lasso(alpha=alpha)

        # Get the coefficients for different values of alpha
        alphas = np.logspace(-5, 2, 100)
        lasso_coefs = []

        for alpha in alphas:
            model.set_params(alpha=alpha)
            model.fit(X, y)
            lasso_coefs.append(model.coef_.flatten())  # Flatten the 2D array to 1D

        # Plot the coefficients against alpha on a log scale
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, lasso_coefs, label="Lasso", color='r')
        plt.xscale('log')
        plt.xlabel('Regularization Strength (alpha)')
        plt.ylabel('Coefficients')
        plt.title('Lasso Regularization Path')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def elastic_net_regression_model(file_path, alpha=1.0):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variable = np.random.choice([col for col in df.columns if col != dependent_variable])

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize the Ridge, Lasso, and Elastic Net regression models
        ridge = Ridge()
        lasso = Lasso()
        elastic_net = ElasticNet()

        # Fit the models to the data
        ridge.fit(X, y)
        lasso.fit(X, y)
        elastic_net.fit(X, y)

        # Get the coefficients for different values of alpha
        alphas = np.logspace(-5, 2, 100)
        ridge_coefs = []
        lasso_coefs = []
        elastic_net_coefs = []

        for alpha in alphas:
            ridge.set_params(alpha=alpha)
            lasso.set_params(alpha=alpha)
            elastic_net.set_params(alpha=alpha)
            
            ridge.fit(X, y)
            lasso.fit(X, y)
            elastic_net.fit(X, y)
            
            ridge_coefs.append(ridge.coef_.flatten())  # Flatten the 2D array to 1D
            lasso_coefs.append(lasso.coef_.flatten())  # Flatten the 2D array to 1D
            elastic_net_coefs.append(elastic_net.coef_.flatten())  # Flatten the 2D array to 1D

        # Plot the coefficients against alpha on a log scale
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, ridge_coefs, label="Ridge", color='b')
        plt.plot(alphas, lasso_coefs, label="Lasso", color='r')
        plt.plot(alphas, elastic_net_coefs, label="Elastic Net", color='g')
        plt.xscale('log')
        plt.xlabel('Regularization Strength (alpha)')
        plt.ylabel('Coefficients')
        plt.title('Regularization Path')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def time_series_regression_model(file_path, lag_order=1):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variable = np.random.choice([col for col in df.columns if col != dependent_variable])

        # Extract features and target variable
        X = df[independent_variable]
        y = df[dependent_variable]

        # Create lagged variables
        for i in range(1, lag_order + 1):
            df[f'{dependent_variable}_lag_{i}'] = df[dependent_variable].shift(i)

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Reindex the arrays to align the indices
        X = X.reindex_like(df)
        y = y.reindex_like(df)

        # Fit the time series regression model
        model = sm.OLS(y, sm.add_constant(df.drop(columns=[dependent_variable])))
        results = model.fit()

        # Print the summary of the model
        print(results.summary())

        # Plot the actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, y, label='Actual', color='blue')
        plt.plot(df.index, results.predict(), label='Predicted', color='red')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Regression')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def random_forest_regression_model(file_path, n_estimators=100, max_depth=None):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizModel.label_encode_non_numeric_columns(df)

        # Randomly select dependent and independent variables
        dependent_variable = np.random.choice(df.columns)
        independent_variable = np.random.choice([col for col in df.columns if col != dependent_variable])

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize and fit the Random Forest regression model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the model
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Random Forest Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Optionally, you can print feature importances
        print("Feature Importances:", model.feature_importances_)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def apply_model(file_path, model_type):
        if model_type == 'linear_regression':
            return AskWhizModel.linear_regression_model(file_path)
        elif model_type == 'multiple_regression':
            return AskWhizModel.multiple_regression_model(file_path)
        elif model_type == 'logistic_regression':
            return AskWhizModel.logistic_regression_model(file_path)
        elif model_type == 'poisson_regression':
            return AskWhizModel.poisson_regression_model(file_path)
        elif model_type == 'negative_binomial_regression':
            return AskWhizModel.negative_binomial_regression_model(file_path)
        elif model_type == 'ridge_regression':
            return AskWhizModel.ridge_regression_model(file_path, alpha=1.0)
        elif model_type == 'elastic_net_regression':
            return AskWhizModel.elastic_net_regression_model(file_path, alpha=1.0)
        elif model_type == 'time_series_regression':
            return AskWhizModel.time_series_regression_model(file_path, lag_order=1)
        elif model_type == 'lasso_regression':
            return AskWhizModel.lasso_regression_model(file_path, alpha=1.0)
        elif model_type == 'random_forest_regression':
            return AskWhizModel.random_forest_regression_model(file_path, n_estimators=100, max_depth=None)
        else:
            return f"Invalid regression type: {model_type}"

class AskWhizTest:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df
    
    @staticmethod
    def t_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Perform t-test for each pair of numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        t_test_results = []

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                variable1_name = numeric_columns[i]
                variable2_name = numeric_columns[j]

                # Extract numerical values for the t-test
                variable1 = df[variable1_name].dropna().values
                variable2 = df[variable2_name].dropna().values

                # Perform t-test
                t_stat, p_value = ttest_ind(variable1, variable2)
                significance_level = 0.05

                # Store the results
                result = f"T-Statistic for {variable1_name} and {variable2_name}: {t_stat}\n"
                result += f"P-Value for {variable1_name} and {variable2_name}: {p_value}\n"
                result += f"Significance for {variable1_name} and {variable2_name}: {'Reject' if p_value < significance_level else 'Fail to reject'}\n"
                t_test_results.append(result)

        # Analyze the overall dataset
        total_tests = len(t_test_results)
        if total_tests > 0:
            significant_tests = sum([1 for result in t_test_results if 'Reject' in result])
            percentage_significant = (significant_tests / total_tests) * 100
            analysis = f"Overall, {significant_tests} out of {total_tests} t-tests ({percentage_significant:.2f}%) show significant differences in means, suggesting that there are significant differences between some of the numeric variables in the dataset."
            t_test_results.append(analysis)
        else:
            analysis = "No t-tests were performed as there are not enough numeric variables in the dataset."
            t_test_results.append(analysis)

        return '\n'.join(t_test_results)

    @staticmethod
    def null_and_hypothesis_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Perform null hypothesis testing and calculate mean
        null_hypothesis_results = []
        for column in df.columns:
            for other_column in df.columns:
                if column != other_column:
                    variable1_name = column
                    variable2_name = other_column

                    # Extract numerical values for the hypothesis test
                    variable1 = df[variable1_name].dropna().values
                    variable2 = df[variable2_name].dropna().values

                    # Perform t-test
                    t_stat, p_value = ttest_ind(variable1, variable2)
                    significance_level = 0.05

                    # Calculate mean
                    mean_variable1 = np.mean(variable1)
                    mean_variable2 = np.mean(variable2)

                    # Store results
                    null_hypothesis_results.append(f"Null hypothesis testing for columns {variable1_name} and {variable2_name}:\n")
                    null_hypothesis_results.append(f"T-Statistic: {t_stat}\n")
                    null_hypothesis_results.append(f"P-Value: {p_value}\n")
                    null_hypothesis_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")
                    null_hypothesis_results.append(f"Mean of {variable1_name}: {mean_variable1}\n")
                    null_hypothesis_results.append(f"Mean of {variable2_name}: {mean_variable2}\n")

        results.append('\n'.join(null_hypothesis_results))

        # Analyze overall test results
        total_tests = len(null_hypothesis_results) // 6
        if total_tests > 0:
            significant_tests = sum([1 for i in range(3, len(null_hypothesis_results), 6) if 'Reject' in null_hypothesis_results[i]])
            percentage_significant = (significant_tests / total_tests) * 100
            analysis = f"Overall, {significant_tests} out of {total_tests} tests ({percentage_significant:.2f}%) show significant differences between means, suggesting that there are significant differences between some of the numeric variables in the dataset."
            results.append(analysis)
        else:
            analysis = "No tests were performed as there are not enough numeric variables in the dataset."
            results.append(analysis)

        return '\n'.join(results)


    @staticmethod
    def anova_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform ANOVA test
        anova_results = []
        significant_variables = set()

        for column in df.columns:
            variable_name = column

            # Extract numerical values for the ANOVA test
            variable = df[variable_name].dropna().values

            # Perform ANOVA test
            if len(variable) > 1:
                f_stat, p_value = f_oneway(variable, np.zeros_like(variable))
                anova_results.append({
                    'Variable': variable_name,
                    'F-Statistic': f_stat,
                    'P-Value': p_value,
                    'Significance': 'Reject' if p_value < significance_level else 'Fail to reject'
                })

                # Check if the test is significant
                if p_value < significance_level:
                    significant_variables.add(variable_name)

            else:
                anova_results.append({
                    'Variable': variable_name,
                    'Insufficient data': True
                })

        # Analyze overall ANOVA test results
        total_variables = len(df.columns)
        significant_variables_count = len(significant_variables)

        if total_variables > 0:
            percentage_significant = (significant_variables_count / total_variables) * 100
            analysis = f"Overall, {significant_variables_count} out of {total_variables} ANOVA tests ({percentage_significant:.2f}%) show significant differences between groups, suggesting that there are significant differences between some of the numeric variables in the dataset."
            results.append(analysis)
            if significant_variables:
                results.append("Variables with significant differences:")
                for variable in significant_variables:
                    results.append(f"- {variable}")
        else:
            analysis = "No ANOVA tests were performed as there are not enough numeric variables with sufficient data in the dataset."
            results.append(analysis)

        # Add detailed test results to the output
        detailed_results = []
        for test_result in anova_results:
            if test_result.get('Insufficient data'):
                detailed_results.append(f"ANOVA test for {test_result['Variable']}:\nInsufficient data for ANOVA test\n")
            else:
                detailed_results.append(f"ANOVA test for {test_result['Variable']}:\n")
                detailed_results.append(f"F-Statistic: {test_result['F-Statistic']}\n")
                detailed_results.append(f"P-Value: {test_result['P-Value']}\n")
                detailed_results.append(f"Significance: {test_result['Significance']}\n")
        results.extend(detailed_results)

        return '\n'.join(results)


    @staticmethod
    def chi_square_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform chi-square test for each pair of columns
        chi_square_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    contingency_table = pd.crosstab(df[column1], df[column2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    chi_square_results.append(f"Chi-square test for columns {column1} and {column2}:\n")
                    chi_square_results.append(f"Chi-square statistic: {chi2}\n")
                    chi_square_results.append(f"P-value: {p_value}\n")
                    chi_square_results.append(f"Degrees of freedom: {dof}\n")
                    chi_square_results.append(f"Expected frequencies:\n{expected}\n")
                    chi_square_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")
                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall chi-square test results
        total_tests = len(chi_square_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} chi-square tests ({percentage_significant:.2f}%) show significant associations between categorical variables, suggesting that there are significant relationships between some pairs of variables in the dataset."
            results.append(analysis)
        else:
            analysis = "No chi-square tests were performed as there are not enough categorical variables with sufficient data in the dataset."
            results.append(analysis)

        results.append('\n'.join(chi_square_results))

        return '\n'.join(results)

    @staticmethod
    def mann_whitney_u_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Mann-Whitney U test for each pair of columns
        mann_whitney_u_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    # Extract the non-missing and non-null values for each column
                    values1 = df[column1].dropna().values
                    values2 = df[column2].dropna().values

                    # Perform Mann-Whitney U test
                    statistic, p_value = mannwhitneyu(values1, values2)

                    mann_whitney_u_results.append(f"Mann-Whitney U test for columns {column1} and {column2}:\n")
                    mann_whitney_u_results.append(f"Mann-Whitney U statistic: {statistic}\n")
                    mann_whitney_u_results.append(f"P-value: {p_value}\n")
                    mann_whitney_u_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall Mann-Whitney U test results
        total_tests = len(mann_whitney_u_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Mann-Whitney U tests ({percentage_significant:.2f}%) show significant differences between distributions."
            results.append(analysis)
        else:
            analysis = "No Mann-Whitney U tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(mann_whitney_u_results))

        return '\n'.join(results)

    @staticmethod
    def ks_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Kolmogorov-Smirnov test for each pair of columns
        ks_test_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    # Perform Kolmogorov-Smirnov test
                    statistic, p_value = ks_2samp(df[column1], df[column2])

                    ks_test_results.append(f"Kolmogorov-Smirnov test for columns {column1} and {column2}:\n")
                    ks_test_results.append(f"Kolmogorov-Smirnov statistic: {statistic}\n")
                    ks_test_results.append(f"P-value: {p_value}\n")
                    ks_test_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall Kolmogorov-Smirnov test results
        total_tests = len(ks_test_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Kolmogorov-Smirnov tests ({percentage_significant:.2f}%) show significant differences between distributions."
            results.append(analysis)
        else:
            analysis = "No Kolmogorov-Smirnov tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(ks_test_results))

        return '\n'.join(results)

    @staticmethod    
    def shapiro_wilk_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Shapiro-Wilk test for each column
        shapiro_test_results = []
        significant_tests = []

        for column in df.columns:
            # Perform Shapiro-Wilk test
            statistic, p_value = shapiro(df[column])

            shapiro_test_results.append(f"Shapiro-Wilk test for column {column}:\n")
            shapiro_test_results.append(f"Shapiro-Wilk statistic: {statistic}\n")
            shapiro_test_results.append(f"P-value: {p_value}\n")
            shapiro_test_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

            if p_value < significance_level:
                significant_tests.append(column)

        # Analyze overall Shapiro-Wilk test results
        total_tests = len(df.columns)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Shapiro-Wilk tests ({percentage_significant:.2f}%) show significant departure from normality."
            if significant_tests:
                analysis += "\nVariables with significant departure from normality:\n"
                for variable in significant_tests:
                    analysis += f"- {variable}\n"
            else:
                analysis += "\nNo variables show significant departure from normality."
            results.append(analysis)
        else:
            analysis = "No Shapiro-Wilk tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(shapiro_test_results))

        return '\n'.join(results)


    @staticmethod
    def durbin_watson_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Create and fit the linear regression model
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions using the model
        y_pred = model.predict(X)

        # Calculate the residuals
        residuals = y - y_pred

        # Perform the Durbin-Watson test
        dw_statistic = durbin_watson(residuals)

        # Calculate critical values for a given significance level
        n = len(residuals)
        k = X.shape[1]
        critical_values = [(0, 2), (2, 1.5 + 0.15 * k), (2, 2 - 0.15 * k), (2, 4)]

        # Analyze the overall test result
        if dw_statistic < 1.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating strong positive autocorrelation."
        elif 1.5 <= dw_statistic < 2.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating no significant autocorrelation."
        elif 2.5 <= dw_statistic < 3.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating weak negative autocorrelation."
        else:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating strong negative autocorrelation."

        # Add critical values and recommendation to the result
        result += "\n\nCritical Values for DW Statistic (5% Significance Level):\n"
        for i, (lower, upper) in enumerate(critical_values, start=1):
            result += f"Region {i}: ({lower}, {upper})\n"
        
        return result

    @staticmethod
    def breusch_godfrey_test(file_path, max_lags=1):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Fit a linear regression model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        # Initialize lists to store results
        bg_test_statistics = []
        bg_p_values = []
        bg_dfs = []
        bg_residuals = []

        # Loop through each independent variable (column) and perform the Breusch-Godfrey test
        for column in X.columns:
            # Fit a linear regression model
            model = sm.OLS(y, sm.add_constant(X[column])).fit()

            # Perform the Breusch-Godfrey test
            bg_test_statistic, bg_p_value, bg_df, bg_residual = acorr_breusch_godfrey(model, nlags=max_lags)

            # Append the results to the lists
            bg_test_statistics.append(bg_test_statistic)
            bg_p_values.append(bg_p_value)
            bg_dfs.append(bg_df)
            bg_residuals.append(bg_residual)

        # Identify which variables have significant serial correlation in their residuals
        significant_results = [(X.columns[i], p_value < 0.05) for i, p_value in enumerate(bg_p_values)]

        # Generate an overall analysis of the test results
        overall_analysis = "Overall Analysis:\n"
        if any(significant for _, significant in significant_results):
            overall_analysis += "Variables with significant serial correlation in the residuals:\n"
            for variable, significant in significant_results:
                if significant:
                    overall_analysis += f"- {variable}\n"
        else:
            overall_analysis += "No variables have significant serial correlation in the residuals.\n"

        # Return the results for each column on separate lines along with the overall analysis
        results = overall_analysis
        for i, column in enumerate(X.columns):
            results += f"\nColumn: {column}\n"
            results += f"Breusch-Godfrey Test Statistic: {bg_test_statistics[i]}\n"
            results += f"Breusch-Godfrey P-Value: {bg_p_values[i]}\n"
            results += f"Breusch-Godfrey Degrees of Freedom: {bg_dfs[i]}\n"
            results += f"Breusch-Godfrey Residuals: {bg_residuals[i]}\n"
            if significant_results[i][1]:
                results += f"Significant serial correlation detected.\n"
            else:
                results += f"No significant serial correlation detected.\n"

        return results

    @staticmethod
    def heteroscedasticity_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Initialize lists to store results
        het_test_statistics = []
        het_p_values = []
        significant_variables = []

        # Loop through each independent variable (column) and perform the Heteroscedasticity Test
        for column in X.columns:
            # Fit a linear regression model
            model = sm.OLS(y, sm.add_constant(X[column])).fit()

            # Perform the Heteroscedasticity Test
            het_test_statistic, het_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)

            # Append the results to the lists
            het_test_statistics.append(het_test_statistic)
            het_p_values.append(het_p_value)

            # Check if the test is significant
            if het_p_value < 0.05:
                significant_variables.append(column)

        # Analyze overall Heteroscedasticity Test results
        total_tests = len(het_test_statistics)
        significant_tests_count = sum(p_value < 0.05 for p_value in het_p_values)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Heteroscedasticity tests ({percentage_significant:.2f}%) show significant results, indicating the presence of heteroscedasticity."
        else:
            analysis = "No Heteroscedasticity tests were performed as there are not enough independent variables in the dataset."

        # Add variables with significant results to the analysis
        if significant_variables:
            analysis += "\nVariables with significant heteroscedasticity:\n"
            for variable in significant_variables:
                analysis += f"- {variable}\n"
        else:
            analysis += "\nNo variables show significant heteroscedasticity."

        # Return the results for each column on separate lines along with the overall analysis
        results = analysis + "\n\n"
        for i, column in enumerate(X.columns):
            results += f"Column: {column}\n"
            results += f"Heteroscedasticity Test Statistic: {het_test_statistics[i]}\n"
            results += f"Heteroscedasticity P-Value: {het_p_values[i]}\n\n"

        return results

    @staticmethod
    def multicollinearity_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = AskWhizTest.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Fit a linear regression model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        # Calculate the Variance Inflation Factor (VIF) for each independent variable
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
        vif["features"] = model.model.exog_names

        # Find the maximum VIF value and its corresponding feature
        max_vif = vif["VIF Factor"].max()
        max_vif_feature = vif.loc[vif["VIF Factor"].idxmax(), "features"]

        # Display the results
        multicollinearity_result = f"Multicollinearity Test\n"
        multicollinearity_result += f"Maximum VIF: {max_vif}\n"
        multicollinearity_result += f"Corresponding Feature: {max_vif_feature}\n"

        # Display multicollinear columns if present
        multicollinear_columns = vif[vif["VIF Factor"] > 10]["features"].tolist()
        if multicollinear_columns:
            multicollinearity_result += f"Multicollinear Columns: {', '.join(multicollinear_columns)}\n"

        # Analyze overall multicollinearity test results
        total_tests = len(vif)
        significant_tests_count = sum(vif["VIF Factor"] > 10)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} variables ({percentage_significant:.2f}%) exhibit multicollinearity, indicating potential issues with the model's stability and interpretability."
            multicollinearity_result += analysis
        else:
            analysis = "No multicollinearity issues detected in the dataset."
            multicollinearity_result += analysis

        return multicollinearity_result
    
# Visualization
class Visualization:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        label_encoders = {}
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
        return df, label_encoders
 
    @staticmethod
    def generate_chart(dataset_path, chart_type, x_label, y_label, z_label):
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)

        original_df = df.copy()  # Keep a copy of the original data for labels
        
        # Label encode non-numeric columns and get mappings
        df, label_encoders = Visualization.label_encode_non_numeric_columns(df)
        
        # Configure Matplotlib to use a non-GUI backend
        plt.switch_backend('agg')
        plt.ioff()

        # plt.figure(figsize=(8, 8))

        # Check the selected chart type and generate the corresponding chart
        if chart_type == 'histogram':
            # Create a figure for the histogram
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the histogram for the selected column
            ax.hist(df[x_label], bins=20, edgecolor='k')

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=90)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'box_plot':
            # Create a figure for the box plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the box plot for the specified x_label
            df[[x_label]].boxplot(ax=ax, vert=False, grid=False)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_yticks(range(len(le.classes_)))
                ax.set_yticklabels(le.classes_)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel('Value')
            ax.set_ylabel(column_name)
            ax.set_title(f'Box Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'violin_plot':
            # Create a figure for the violin plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the violin plot for the specified x_label
            sns.violinplot(data=df[[x_label]], ax=ax)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=90)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Value')
            ax.set_title(f'Violin Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'dot_plot':
            # Create a figure for the dot plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the dot plot for the specified x_label
            sns.stripplot(data=df[[x_label]], ax=ax, jitter=True)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=90)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Value')
            ax.set_title(f'Dot Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'quantile-quantile_plot':
            # Create a figure for the Q-Q plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the Q-Q plot for the specified x_label
            sm.qqplot(df[x_label], line='45', ax=ax)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=45)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title(f'Q-Q Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'density_plot':
            # Create a figure for the density plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the density plot for the specified x_label
            sns.kdeplot(data=df[x_label], ax=ax)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=45)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Density Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'distribution_plot':
            # Create a figure for the distribution plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the distribution plot for the specified x_label
            sns.histplot(df[x_label], kde=True, ax=ax, stat='density')

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=45)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution Plot for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'scatter_plot':
            # Create a figure for the scatter plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the scatter plot for the specified x_label and y_label
            ax.scatter(df[x_label], df[y_label], alpha=0.5)

            # Use original labels for categorical data if available
            if label_encoders and x_label in label_encoders:
                le = label_encoders[x_label]
                ax.set_xticks(range(len(le.classes_)))
                ax.set_xticklabels(le.classes_, rotation=45)

            if label_encoders and y_label in label_encoders:
                le = label_encoders[y_label]
                ax.set_yticks(range(len(le.classes_)))
                ax.set_yticklabels(le.classes_)

            # Set labels and title
            x_column_name = original_df[x_label].name if original_df is not None else x_label
            y_column_name = original_df[y_label].name if original_df is not None else y_label
            ax.set_xlabel(x_column_name)
            ax.set_ylabel(y_column_name)
            ax.set_title(f'Scatter Plot for {x_column_name} vs {y_column_name}')

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()
        elif chart_type == 'line_plot':
            # Create a figure for the line plot
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Plot the line plot for the specified x_label and y_label
            df.plot(x=x_label, y=y_label, ax=ax)

            # Use original labels for categorical data if available
            if label_encoders:
                if x_label in label_encoders:
                    le_x = label_encoders[x_label]
                    ax.set_xticks(range(len(le_x.classes_)))
                    ax.set_xticklabels(le_x.classes_, rotation=90)

                if y_label in label_encoders:
                    le_y = label_encoders[y_label]
                    ax.set_yticks(range(len(le_y.classes_)))
                    ax.set_yticklabels(le_y.classes_, rotation=90)

            # Set labels and title
            x_column_name = original_df[x_label].name if original_df is not None else x_label
            y_column_name = original_df[y_label].name if original_df is not None else y_label
            ax.set_xlabel(x_column_name)
            ax.set_ylabel(y_column_name)
            ax.set_title(f'Line Plot for {y_column_name} against {x_column_name}')

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()
        elif chart_type == 'bar_chart':
            # Create a figure for the bar chart
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Check if the column is numeric and if the values are large
            if pd.api.types.is_numeric_dtype(df[x_label]) and df[x_label].nunique() > 10:
                # Create bins
                df['binned'] = pd.cut(df[x_label], bins=10)
                category_counts = df['binned'].value_counts().sort_index()
                sns.barplot(x=category_counts.index.astype(str), y=category_counts.values, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                # Plot the bar chart for the specified x_label
                category_counts = df[x_label].value_counts()
                sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)

                # Use original labels for categorical data if available
                if label_encoders and x_label in label_encoders:
                    le = label_encoders[x_label]
                    ax.set_xticks(range(len(le.classes_)))
                    ax.set_xticklabels(le.classes_, rotation=90)

            # Set labels and title
            column_name = original_df[x_label].name if original_df is not None else x_label
            ax.set_xlabel(column_name)
            ax.set_ylabel('Count')
            ax.set_title(f'Bar Chart for {column_name}')

            # Adjust layout
            plt.tight_layout()
            plt.show()
        elif chart_type == 'pie_chart':
            # Create a figure for the pie chart
            plt.figure(figsize=(5, 4.5))

            # Handle categorical or numeric data
            if x_label in label_encoders:
                # Handle categorical columns
                data = df[x_label][df[x_label] >= 0]  # Skip negative values
                counts = data.value_counts()
                le = label_encoders[x_label]
                labels = le.inverse_transform(counts.index)
            else:
                # Handle numeric columns with bins
                data = df[x_label][df[x_label] >= 0]  # Skip negative values
                bins = pd.cut(data, bins=5)
                counts = bins.value_counts()
                labels = counts.index.astype(str)

            # Generate pie chart
            plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title(f'Pie Chart for {original_df[x_label].name}')

            # Show the plot
            # plt.tight_layout()
            plt.show()
        elif chart_type == 'correlation_matrix':
            corr_matrix = df[[x_label, y_label]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title(f'Correlation Matrix between {x_label} and {y_label}')
            plt.show()
        elif chart_type == 'heatmap':
            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(5, 4.5))

            # Generate a custom colormap
            cmap = sns.color_palette("coolwarm", as_cmap=True)

            # Draw the heatmap with the correct aspect ratio
            sns.heatmap(df.corr(), cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f')

            # Set original labels for the axes
            labels = []
            for column in df.columns:
                if column in label_encoders:
                    labels.append(original_df[column].name)
                else:
                    labels.append(original_df[column].name)

            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

            plt.title('Heatmap of Dataset')
            plt.tight_layout()
            plt.show()
        elif chart_type == '3d_scatter_plot':
            # Set up the figure and 3D axis
            fig = plt.figure(figsize=(5, 4.5))
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D scatter plot
            ax.scatter(df[x_label], df[y_label], df[z_label], c='b', marker='o')

            # Use original labels for categorical data if available
            if label_encoders:
                if x_label in label_encoders:
                    le_x = label_encoders[x_label]
                    ax.set_xticks(range(len(le_x.classes_)))
                    ax.set_xticklabels(le_x.classes_, rotation=90)

                if y_label in label_encoders:
                    le_y = label_encoders[y_label]
                    ax.set_yticks(range(len(le_y.classes_)))
                    ax.set_yticklabels(le_y.classes_, rotation=90)

                if z_label in label_encoders:
                    le_z = label_encoders[z_label]
                    ax.set_zticks(range(len(le_z.classes_)))
                    ax.set_zticklabels(le_z.classes_, rotation=90)

            # Set labels and title
            x_column_name = original_df[x_label].name if original_df is not None else x_label
            y_column_name = original_df[y_label].name if original_df is not None else y_label
            z_column_name = original_df[z_label].name if original_df is not None else z_label
            ax.set_xlabel(x_column_name)
            ax.set_ylabel(y_column_name)
            ax.set_zlabel(z_column_name)
            ax.set_title(f'3D Scatter Plot of {x_column_name}, {y_column_name}, and {z_column_name}')

            # Show the plot
            plt.tight_layout()
            plt.show()
        elif chart_type == 'parallel_coordinate_plot':
            # Create parallel coordinate plot
            plt.figure(figsize=(5, 4.5))
            parallel_coordinates(df, df.columns[-1], colormap='viridis')

            # Rotate x-axis labels
            plt.xticks(rotation=90)

            # Turn off grid and legend
            plt.grid(False)
            plt.legend().set_visible(False)

            # Set title
            plt.title('Parallel Coordinate Plot')

            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()
        elif chart_type == 'andrews_plot':
            plt.subplots_adjust(top=0.9)
            plt.figure(figsize=(5, 4.5))
            andrews_curves(df, df.columns[0], colormap='viridis')
            plt.grid(False)
            plt.legend().set_visible(False)
            plt.xlabel('Andrews Curve')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.title('Andrews Plot')
        elif chart_type == 'radar_chart':
            num_vars = df.shape[1]
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            values = df.values
            values = np.concatenate((values, values[:, [0]]), axis=1)
            angles += angles[:1]
            plt.figure(figsize=(5, 4.5))
            for i in range(len(values)):
                plt.polar(angles, values[i], marker='o', label=f'{df.index[i]}')
            plt.legend().set_visible(False)
            plt.thetagrids(np.degrees(angles[:-1]), df.columns)
            plt.tight_layout()
            plt.title('Radar Chart')
        elif chart_type == 'chernoff_faces':
            normalized_df = (df - df.min()) / (df.max() - df.min())
            max_faces_to_display = 16
            num_faces = min(len(df), max_faces_to_display)
            faces_per_row = 4
            num_rows = int(np.ceil(num_faces / faces_per_row))
            fig, ax = plt.subplots(num_rows, faces_per_row, figsize=(7, 2 * num_rows))
            for i in range(num_faces):
                row_idx = i // faces_per_row
                col_idx = i % faces_per_row
                params = {
                    'head_radius': normalized_df.iloc[i, 0],
                    'eye_size': normalized_df.iloc[i, 1] if df.shape[1] > 1 else 0.1,
                    'mouth_width': normalized_df.iloc[i, 2] if df.shape[1] > 2 else 0.5,
                    'mouth_smile': normalized_df.iloc[i, 3] if df.shape[1] > 3 else 0.5,
                }
                ax[row_idx, col_idx].set_xlim(-1.5, 1.5)
                ax[row_idx, col_idx].set_ylim(-1.5, 1.5)
                ax[row_idx, col_idx].axis('off')
                circle_head = plt.Circle((0, 0), params['head_radius'], color='yellow', fill=True, ec='black')
                ax[row_idx, col_idx].add_artist(circle_head)
                circle_eye_left = plt.Circle((-0.4, 0.3), params['eye_size'], color='black', fill=True)
                circle_eye_right = plt.Circle((0.4, 0.3), params['eye_size'], color='black', fill=True)
                ax[row_idx, col_idx].add_artist(circle_eye_left)
                ax[row_idx, col_idx].add_artist(circle_eye_right)
                mouth_x = np.linspace(-0.4, 0.4, 100)
                mouth_y = params['mouth_smile'] * np.sin(np.pi * mouth_x / params['mouth_width'])
                ax[row_idx, col_idx].plot(mouth_x, mouth_y, color='black')
            for i in range(num_faces, num_rows * faces_per_row):
                row_idx = i // faces_per_row
                col_idx = i % faces_per_row
                fig.delaxes(ax[row_idx, col_idx])
            plt.suptitle('Chernoff Faces')
            plt.tight_layout()
        elif chart_type == 'tree_map':
            # Sort values to prioritize larger values
            df = df.sort_values(by=df.columns[1], ascending=False)
            
            # Take the top N values to display in the tree map
            N = 200  
            df_top = df.head(N)
            
            # Replace 0 values with a small non-zero value to avoid errors
            df_top.iloc[:, 1] = df_top.iloc[:, 1].replace(0, 1)
            
            # Generate the tree map
            plt.figure(figsize=(5, 4.5))
            squarify.plot(sizes=df_top.iloc[:, 1], label=df_top.iloc[:, 0], alpha=0.8)
            plt.title('Tree Map')
            plt.tight_layout()
            plt.axis('off')
            plt.show()
        elif chart_type == 'sunburst_chart':
            # Sort values to prioritize larger values
            df = df.sort_values(by=df.columns[1], ascending=False)

            # Take the top N values to display in the tree map
            N = 100 
            df_top = df.head(N)

            # Replace 0 values with a small non-zero value to avoid errors
            df_top.iloc[:, 1] = df_top.iloc[:, 1].replace(0, 1)

            # Calculate angles and radii for sectors
            values = df_top.iloc[:, 1].values
            labels = df_top.iloc[:, 0].values
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
            radii = np.sqrt(values / np.pi)

            # Define a color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

            # Create subplots and plot the sectors with different colors
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 4.5))
            bars = ax.bar(angles, radii, width=(2*np.pi) / len(values), bottom=0.0, color=colors, edgecolor='black', alpha=0.8)

            # Add labels to each sector with adjusted vertical position
            label_distance = 1.5  # Adjust the distance of labels from the bars
            for bar, angle, label in zip(bars, angles, labels):
                ax.text(angle, bar.get_height() + label_distance, label, ha='center', va='bottom', fontsize=8)

            ax.set_xticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.title('Sunburst Chart')
            plt.show()
        elif chart_type == 'force_directed_graph':
            # Create a figure and adjust its size
            plt.figure(figsize=(5, 4.5))

            # Create an empty graph
            G = nx.Graph()

            # Add nodes for each column in the DataFrame
            for column in df.columns:
                G.add_node(column)

            # Add edges between each pair of columns
            for column1 in df.columns:
                for column2 in df.columns:
                    if column1 != column2:
                        G.add_edge(column1, column2)

            # Define the layout for the graph (spring layout)
            pos = nx.spring_layout(G)

            # Draw the graph with labels and styling
            nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', edge_color='gray', linewidths=1, font_size=8)

            # Set the title and adjust layout
            plt.title('Force Directed Graph')
            plt.tight_layout()

            # Show the plot
            plt.show()
        elif chart_type == 'sankey_diagram':
            # Define your data
            flows = []  # List to store flows

            # Iterate over columns of the DataFrame
            for i in range(df.shape[1] - 1):
                # Add flows for each pair of columns
                flows.extend([1, -1])

            # Create the Sankey diagram
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sankey = Sankey(ax=ax, unit=None)
            sankey.add(flows=flows, orientations=[0, 1] * (len(flows) // 2))
            
            sankey.finish()

            # Set title and adjust layout
            plt.title('Sankey Diagram')
            plt.tight_layout()
            plt.show()
        elif chart_type == 'covariance_matrix':
            plt.figure(figsize=(5, 4.5))
            cov_matrix = df.cov()
            sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Covariance Matrix')
            plt.tight_layout()
            plt.show()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
# Analysis
class Analysis:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            # Convert values to strings before label encoding
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df

    @staticmethod
    def perform_eda(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        eda_result = f"Exploratory Data Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Summary statistics for all numerical columns
        selected_numerical_df = df.select_dtypes(include=['int64', 'float64'])

        if not selected_numerical_df.empty:
            eda_result += "Summary Statistics for All Numerical Columns:\n"
            eda_result += selected_numerical_df.describe().to_string() + "\n"
            eda_result += "\n"

            # Info about data types and missing values for all variables
            eda_result += "Data Types and Missing Values for All Variables:\n"
            selected_info_str = selected_numerical_df.info()
            eda_result += selected_info_str if selected_info_str is not None else "No missing values\n"
            eda_result += "\n"

            # Univariate Analysis - Count of unique values for each column
            eda_result += "Count of Unique Values for Each Column:\n"
            eda_result += selected_numerical_df.nunique().to_string() + "\n"
            eda_result += "\n"

            # Correlation matrix for all numerical columns
            eda_result += "Correlation Matrix for All Numerical Columns:\n"
            eda_result += selected_numerical_df.corr().to_string() + "\n"

        else:
            eda_result += "No numeric columns in the dataset.\n"

        # Data Cleaning and Preprocessing
        eda_result += "Data Cleaning and Preprocessing:\n"
        # Handle missing values
        df_cleaned = df.dropna()
        eda_result += f"Dataset after removing rows with missing values: {len(df_cleaned)} rows\n\n"

        return eda_result

    @staticmethod
    def perform_descriptive_statistics(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        desc_stats_result = f"Descriptive Statistics for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Summary statistics for the entire dataset
        desc_stats_result += df.describe().to_string()

        # Skewness and Kurtosis for each numerical column
        desc_stats_result += "\nSkewness:\n"
        desc_stats_result += df.skew().to_string()

        desc_stats_result += "\n\nKurtosis:\n"
        desc_stats_result += df.kurtosis().to_string()

        # Counts and percentages for categorical columns
        desc_stats_result += "\n\nCategorical Column Statistics:\n"
        for column in df.select_dtypes(include='object'):
            desc_stats_result += f"\n{column}:\n"
            desc_stats_result += df[column].value_counts().to_string() + "\n"

        # Frequency distribution for numerical columns (histograms)
        desc_stats_result += "\n\nNumerical Column Frequency Distributions:\n"
        for column in df.select_dtypes(include=['int64', 'float64']):
            desc_stats_result += f"\n{column}:\n"
            desc_stats_result += np.histogram(df[column], bins='auto')[0].tolist().__str__() + "\n"


        return desc_stats_result

    @staticmethod
    def perform_inferential_statistics(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        inferential_stats_result = f"Inferential Statistics for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Calculate confidence intervals for all numeric columns
        for column in df.select_dtypes(include=['number']).columns:
            # Assuming the variable is numerical
            mean = np.mean(df[column])
            std_dev = np.std(df[column], ddof=1)  # ddof=1 for sample standard deviation
            sample_size = len(df[column])

            confidence_level = 0.95
            margin_of_error = norm.ppf((1 + confidence_level) / 2) * (std_dev / np.sqrt(sample_size))
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error

            inferential_stats_result += f"\nConfidence Interval for the Mean of {column}:\n"
            inferential_stats_result += f"({lower_bound}, {upper_bound})\n"

            # Additional information about the normal distribution
            inferential_stats_result += f"Variable {column} follows a normal distribution.\n"
            inferential_stats_result += f"Probability Density Function (PDF) for {column}:\n"
            x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
            pdf = norm.pdf(x, mean, std_dev)
            inferential_stats_result += f"PDF: {pdf}\n"

        return inferential_stats_result

    @staticmethod
    def perform_hypothesis_testing(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        hypothesis_testing_result = f"Hypothesis Testing for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        numeric_columns = df.select_dtypes(include=['number']).columns

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                variable1_name = numeric_columns[i]
                variable2_name = numeric_columns[j]

                # Extract numerical values for the hypothesis test
                variable1 = df[variable1_name].dropna().values
                variable2 = df[variable2_name].dropna().values

                t_stat, p_value = ttest_ind(variable1, variable2)
                significance_level = 0.05

                hypothesis_testing_result += f"\nT-Statistic for Equality of Means ({variable1_name}, {variable2_name}): {t_stat}\n"
                hypothesis_testing_result += f"P-Value for Equality of Means: {p_value}\n"

                if p_value < significance_level:
                    hypothesis_testing_result += "Reject the null hypothesis for Equality of Means\n"
                else:
                    hypothesis_testing_result += "Fail to reject the null hypothesis for Equality of Means\n"

        # Analyze dataset based on the findings
        if len(numeric_columns) > 0:
            analysis = "The hypothesis testing results indicate whether there are significant differences in the means of numeric columns in the dataset. A rejection of the null hypothesis suggests that there is evidence to support differences in means, while a failure to reject the null hypothesis indicates no significant differences.\n\n"
        else:
            analysis = "No numeric columns were found in the dataset, limiting the application of hypothesis testing for comparing means. Consider exploring alternative analytical methods or acquiring additional data.\n\n"

        hypothesis_testing_result += "Analysis of Dataset:\n"
        hypothesis_testing_result += analysis

        return hypothesis_testing_result

    
    @staticmethod
    def perform_correlation_analysis(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        correlation_analysis_result = f"Correlation Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Calculate the correlation matrix for all numeric columns
        correlation_matrix = df.select_dtypes(include=['number']).corr()

        correlation_analysis_result += f"\nCorrelation Matrix:\n{correlation_matrix}\n"

        # Identify significant correlations (you can customize the threshold)
        significance_level = 0.05
        significant_correlations = correlation_matrix[(correlation_matrix.abs() > significance_level) & (correlation_matrix < 1)]
        
        correlation_analysis_result += f"\nSignificant Correlations (p-value < {significance_level}):\n{significant_correlations}\n"

        # Analyze dataset based on the findings
        if significant_correlations.empty:
            analysis = "The correlation analysis did not find any significant correlations between numeric columns in the dataset. This suggests that there may be no strong linear relationships among the variables, or the sample size might be insufficient to detect significant correlations.\n\n"
        else:
            analysis = "The correlation analysis identified significant correlations between some numeric columns in the dataset. This indicates potential linear relationships among these variables, which may provide insights for further analysis or modeling.\n\n"

        correlation_analysis_result += "Analysis of Dataset:\n"
        correlation_analysis_result += analysis

        return correlation_analysis_result

    @staticmethod
    def perform_regression_analysis(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Assuming you have a dependent variable 'y' and independent variables 'x1', 'x2', ...
        dependent_variable = df.columns[0]  # Assuming the first column is the dependent variable
        independent_variables = df.columns[1:]  # Assuming the rest of the columns are independent variables

        # Add a constant term to the independent variables (required for statsmodels)
        X = sm.add_constant(df[independent_variables])

        # Fit the multiple regression model
        model = sm.OLS(df[dependent_variable], X).fit()

        # Get the regression results
        regression_results = model.summary()

        # Analyze the regression results
        if model.pvalues[1:].max() < 0.05:
            analysis = "The regression analysis indicates that at least one independent variable has a significant impact on the dependent variable. This suggests that the model may be useful for predicting or explaining the variation in the dependent variable.\n\n"
        else:
            analysis = "The regression analysis does not show any independent variables with a significant impact on the dependent variable. This suggests that the model may not be useful for predicting or explaining the variation in the dependent variable.\n\n"

        regression_analysis_result = f"Regression Analysis for {file_path}:\n"
        regression_analysis_result += str(regression_results) + "\n\n"  # Convert Summary object to string
        regression_analysis_result += "Analysis of Regression Results:\n"
        regression_analysis_result += analysis

        return regression_analysis_result


    @staticmethod
    def perform_time_series_analysis(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        time_series_analysis_result = f"Time Series Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Assuming you have a timestamp column named 'timestamp'
        if 'timestamp' in df.columns:
            # Convert 'timestamp' column to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Set 'timestamp' as the index
            df.set_index('timestamp', inplace=True)

            time_series_analysis_result += f"\nSummary Statistics Over Time:\n"

            # Calculate summary statistics for each numeric column
            summary_statistics = df.select_dtypes(include=['number']).describe()

            time_series_analysis_result += f"{summary_statistics}\n"

        else:
            time_series_analysis_result += "No timestamp column found in the dataset for time series analysis.\n"

        return time_series_analysis_result

    @staticmethod
    def perform_clustering(file_path):
        num_clusters = 10
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        clustering_result = f"Clustering Analysis for {file_path}:\n"

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Select numeric columns for clustering
        numeric_columns = df.select_dtypes(include=['number']).columns
        data_for_clustering = df[numeric_columns]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)

        # Display the cluster distribution
        cluster_distribution = df['cluster'].value_counts().sort_index()
        clustering_result += f"\nCluster Distribution:\n{cluster_distribution}\n"

        # Analyze the clustering result
        if len(cluster_distribution) > 1:
            clustering_result += "The dataset has been successfully clustered into multiple groups, indicating the presence of distinct patterns or subgroups within the data.\n"
        else:
            clustering_result += "The dataset could not be effectively clustered into multiple groups, suggesting homogeneity or lack of distinguishable patterns within the data.\n"

        return clustering_result

    
    @staticmethod
    def perform_classification(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        classification_result = f"Classification Analysis for {file_path}:\n"

        # Assume the first column as the target variable
        target_column = df.columns[0]

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Select features and target variable
        features = df.drop(columns=[target_column])
        target = df[target_column]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Initialize and train a Random Forest Classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_str = classification_report(y_test, y_pred)

        classification_result += f"\nAccuracy: {accuracy:.2f}\n"
        classification_result += f"\nClassification Report:\n{classification_report_str}\n"

        # Analyze the classification result
        if accuracy > 0.8:
            classification_result += "The classifier achieved high accuracy, indicating that it can effectively distinguish between different classes in the dataset.\n"
            classification_result += "This suggests that the features selected for classification contain significant information for predicting the target variable.\n"
        else:
            classification_result += "The classifier achieved relatively low accuracy, suggesting that it may struggle to accurately classify instances in the dataset.\n"
            classification_result += "This could indicate either a lack of discriminatory power in the features or a need for more sophisticated modeling techniques.\n"

        return classification_result

    @staticmethod
    def perform_dimensionality_reduction(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Analysis.label_encode_non_numeric_columns(df)

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Perform PCA
        pca = PCA()  # No need to specify n_components
        X_pca = pca.fit_transform(X_scaled)

        # Extract the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Calculate the cumulative explained variance
        cumulative_explained_variance = explained_variance_ratio.cumsum()

        # Extract the principal component loadings
        principal_component_loadings = pca.components_

        # Generate textual results
        result = f"Dimensionality Reduction Analysis for {file_path}:\n"
        result += f"\nExplained Variance Ratio: {explained_variance_ratio}\n"
        result += f"\nCumulative Explained Variance: {cumulative_explained_variance}\n"
        result += f"\nPrincipal Component Loadings:\n{principal_component_loadings}\n"

        # Analyze the dimensionality reduction result
        if len(explained_variance_ratio) > 1 and cumulative_explained_variance[-1] > 0.8:
            result += "The dimensionality reduction suggests that a significant portion of the variance in the dataset can be explained by a reduced number of principal components.\n"
            result += "This indicates that the original high-dimensional dataset may contain redundant or unnecessary features.\n"
        else:
            result += "The dimensionality reduction did not result in a substantial reduction in the number of dimensions or explained variance.\n"
            result += "This suggests that the dataset may not benefit significantly from dimensionality reduction techniques like PCA.\n"

        return result

# Regression Model
class RegressionModel:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            # Convert values to strings before label encoding
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df

    @staticmethod
    # Linear Regression Model
    def linear_regression_model(file_path, dependent_variable, independent_variables):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract all columns as independent variables
        independent_variables = df.columns[df.columns != dependent_variable]

        # Create and fit the linear regression model
        X = df[independent_variables]
        y = df[dependent_variable]
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions using the model
        y_pred = model.predict(X)

        # Optionally, you can print coefficients and intercept
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)

        # Create a scatter plot with the regression line
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Linear Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    # Multiple Regression Model
    def multiple_regression_model(file_path, independent_variable, independent_variable2, dependent_variable):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Create and fit the multiple linear regression models
        X1 = df[independent_variable].values.reshape(-1, 1)
        X2 = df[independent_variable2].values.reshape(-1, 1)
        y = df[dependent_variable].values.reshape(-1, 1)
        
        model1 = LinearRegression()
        model1.fit(X1, y)

        model2 = LinearRegression()
        model2.fit(X2, y)

        # Make predictions using the models
        y_pred1 = model1.predict(X1)
        y_pred2 = model2.predict(X2)

        # Create a scatter plot
        plt.figure(figsize=(8, 7))
        plt.scatter(X1, y, color='black', label=f'{independent_variable} vs {dependent_variable}')
        plt.scatter(X2, y, color='red', label=f'{independent_variable2} vs {dependent_variable}')

        # Plot the regression lines
        plt.plot(X1, y_pred1, color='blue', linewidth=3, label=f'Regression Line for {independent_variable}')
        plt.plot(X2, y_pred2, color='green', linewidth=3, label=f'Regression Line for {independent_variable2}')

        # Customize the plot
        plt.title('Multiple Linear Regression')
        plt.xlabel('Independent Variables')
        plt.ylabel('Dependent Variable')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    # Logistic Regression Model
    def logistic_regression_model(file_path, independent_variable, dependent_variable):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Check if the provided column names exist in the DataFrame
        if independent_variable not in df.columns or dependent_variable not in df.columns:
            return f"Error: Column '{independent_variable}' or '{dependent_variable}' not found in the dataset."

        # Extract the relevant columns
        X = df[[independent_variable]]
        y = df[dependent_variable]

        # Create and fit the logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # Create a scatter plot with logistic regression curve using seaborn
        sns.regplot(x=X[independent_variable], y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})

        # Customize the plot
        plt.title('Logistic Regression')
        plt.xlabel(independent_variable)
        plt.ylabel(dependent_variable)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    # Poisson Regression Model
    def poisson_regression_model(file_path, independent_variable, dependent_variable):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract the relevant columns
        X = df[independent_variable].astype(float)
        y = df[dependent_variable].astype(float)

        # Add a constant term to the independent variable
        X = sm.add_constant(X)

        # Fit the Poisson regression model
        model = sm.GLM(y, X, family=sm.families.Poisson())
        model_result = model.fit()

        # Display the summary of the model
        print(model_result.summary())

        # Plot the observed and predicted values
        plt.scatter(X[independent_variable], y, color='black', label='Observed')
        plt.scatter(X[independent_variable], model_result.predict(), color='red', label='Predicted')
        plt.title('Poisson Regression')
        plt.xlabel(independent_variable)
        plt.ylabel(dependent_variable)
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    # Negative Binomial Regression Model
    def negative_binomial_regression_model(file_path, independent_variable, dependent_variable):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract the relevant columns
        X = df[[independent_variable]]
        y = df[dependent_variable]

        # Add a constant term to the independent variable
        X = sm.add_constant(X)

        # Fit the Negative Binomial regression model
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        model_result = model.fit()

        # Display the summary of the model
        print(model_result.summary())

        # Plot observed vs. predicted counts
        predicted_counts = model_result.predict(X)
        observed_counts = y

        plt.bar(range(len(observed_counts)), observed_counts, color='blue', alpha=0.7, label='Observed')
        plt.bar(range(len(predicted_counts)), predicted_counts, color='red', alpha=0.7, label='Predicted')
        plt.title('Negative Binomial Regression')
        plt.xlabel('Observations')
        plt.ylabel('Counts')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Convert the BytesIO buffer to a base64 string
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def ridge_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize and fit the Ridge regression model
        model = Ridge(alpha=alpha)
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the model
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Ridge Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Optionally, you can print coefficients and intercept
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def elastic_net_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0, l1_ratio=0.5):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize the Ridge, Lasso, and Elastic Net regression models
        ridge = Ridge()
        lasso = Lasso()
        elastic_net = ElasticNet()

        # Fit the models to the data
        ridge.fit(X, y)
        lasso.fit(X, y)
        elastic_net.fit(X, y)

        # Get the coefficients for different values of alpha
        alphas = np.logspace(-5, 2, 100)
        ridge_coefs = []
        lasso_coefs = []
        elastic_net_coefs = []

        for alpha in alphas:
            ridge.set_params(alpha=alpha)
            lasso.set_params(alpha=alpha)
            elastic_net.set_params(alpha=alpha)
            
            ridge.fit(X, y)
            lasso.fit(X, y)
            elastic_net.fit(X, y)
            
            ridge_coefs.append(ridge.coef_.flatten())  # Flatten the 2D array to 1D
            lasso_coefs.append(lasso.coef_.flatten())  # Flatten the 2D array to 1D
            elastic_net_coefs.append(elastic_net.coef_.flatten())  # Flatten the 2D array to 1D

        # Plot the coefficients against alpha on a log scale
        plt.figure(figsize=(8, 7))
        plt.plot(alphas, ridge_coefs, label="Ridge", color='b')
        plt.plot(alphas, lasso_coefs, label="Lasso", color='r')
        plt.plot(alphas, elastic_net_coefs, label="Elastic Net", color='g')
        plt.xscale('log')
        plt.xlabel('Regularization Strength (alpha)')
        plt.ylabel('Coefficients')
        plt.title('Regularization Path')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def time_series_regression_model(file_path, dependent_variable, independent_variable, lag_order=1):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract features and target variable
        X = df[independent_variable]
        y = df[dependent_variable]

        # Create lagged variables
        for i in range(1, lag_order + 1):
            df[f'{dependent_variable}_lag_{i}'] = df[dependent_variable].shift(i)

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Reindex the arrays to align the indices
        X = X.reindex_like(df)
        y = y.reindex_like(df)

        # Fit the time series regression model
        model = sm.OLS(y, sm.add_constant(df.drop(columns=[dependent_variable])))
        results = model.fit()

        # Print the summary of the model
        print(results.summary())

        # Plot the actual vs. predicted values
        plt.figure(figsize=(8, 7))
        plt.plot(df.index, y, label='Actual', color='blue')
        plt.plot(df.index, results.predict(), label='Predicted', color='red')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Regression')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def lasso_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize the Lasso regression model
        model = Lasso(alpha=alpha)

        # Get the coefficients for different values of alpha
        alphas = np.logspace(-5, 2, 100)
        lasso_coefs = []

        for alpha in alphas:
            model.set_params(alpha=alpha)
            model.fit(X, y)
            lasso_coefs.append(model.coef_.flatten())  # Flatten the 2D array to 1D

        # Plot the coefficients against alpha on a log scale
        plt.figure(figsize=(8, 7))
        plt.plot(alphas, lasso_coefs, label="Lasso", color='r')
        plt.xscale('log')
        plt.xlabel('Regularization Strength (alpha)')
        plt.ylabel('Coefficients')
        plt.title('Lasso Regularization Path')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data
    
    @staticmethod
    def random_forest_regression_model(file_path, dependent_variable, independent_variable, n_estimators=100, max_depth=None):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = RegressionModel.label_encode_non_numeric_columns(df)

        # Extract features and target variable
        X = df[dependent_variable].values.reshape(-1, 1)  
        y = df[independent_variable].values.reshape(-1, 1)

        # Initialize and fit the Random Forest regression model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the model
        plt.scatter(y, y_pred, color='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='-', color='blue', linewidth=3)

        # Customize the plot
        plt.title('Random Forest Regression')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # Optionally, you can print feature importances
        print("Feature Importances:", model.feature_importances_)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return plot_data

    @staticmethod
    def apply_model(file_path, model_type, independent_variable, independent_variable2, dependent_variable):
        if model_type == 'linear_regression':
            return RegressionModel.linear_regression_model(file_path, independent_variable, dependent_variable)
        elif model_type == 'multiple_regression':
            return RegressionModel.multiple_regression_model(file_path, independent_variable, independent_variable2, dependent_variable)
        elif model_type == 'logistic_regression':
            return RegressionModel.logistic_regression_model(file_path, independent_variable, dependent_variable)
        elif model_type == 'poisson_regression':
            return RegressionModel.poisson_regression_model(file_path, independent_variable, dependent_variable)
        elif model_type == 'negative_binomial_regression':
            return RegressionModel.negative_binomial_regression_model(file_path, independent_variable, dependent_variable)
        elif model_type == 'ridge_regression':
            return RegressionModel.ridge_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0)
        elif model_type == 'elastic_net_regression':
            return RegressionModel.elastic_net_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0, l1_ratio=0.5)
        elif model_type == 'time_series_regression':
            return RegressionModel.time_series_regression_model(file_path, dependent_variable, independent_variable, lag_order=1)
        elif model_type == 'lasso_regression':
            return RegressionModel.lasso_regression_model(file_path, dependent_variable, independent_variable, alpha=1.0)
        elif model_type == 'random_forest_regression':
            return RegressionModel.random_forest_regression_model(file_path, dependent_variable, independent_variable, n_estimators=100, max_depth=None)
        else:
            return f"Invalid regression type: {model_type}"
        
class Test:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            # Convert values to strings before label encoding
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df

    @staticmethod
    def perform_t_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Perform t-test for each pair of numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        t_test_results = []

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                variable1_name = numeric_columns[i]
                variable2_name = numeric_columns[j]

                # Extract numerical values for the t-test
                variable1 = df[variable1_name].dropna().values
                variable2 = df[variable2_name].dropna().values

                # Perform t-test
                t_stat, p_value = ttest_ind(variable1, variable2)
                significance_level = 0.05

                # Store the results
                result = f"T-Statistic for {variable1_name} and {variable2_name}: {t_stat}\n"
                result += f"P-Value for {variable1_name} and {variable2_name}: {p_value}\n"
                result += f"Significance for {variable1_name} and {variable2_name}: {'Reject' if p_value < significance_level else 'Fail to reject'}\n"
                t_test_results.append(result)

        # Analyze the overall dataset
        total_tests = len(t_test_results)
        if total_tests > 0:
            significant_tests = sum([1 for result in t_test_results if 'Reject' in result])
            percentage_significant = (significant_tests / total_tests) * 100
            analysis = f"Overall, {significant_tests} out of {total_tests} t-tests ({percentage_significant:.2f}%) show significant differences in means, suggesting that there are significant differences between some of the numeric variables in the dataset."
            t_test_results.append(analysis)
        else:
            analysis = "No t-tests were performed as there are not enough numeric variables in the dataset."
            t_test_results.append(analysis)

        return '\n'.join(t_test_results)

    @staticmethod
    def perform_null_and_hypothesis_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Perform null hypothesis testing and calculate mean
        null_hypothesis_results = []
        for column in df.columns:
            for other_column in df.columns:
                if column != other_column:
                    variable1_name = column
                    variable2_name = other_column

                    # Extract numerical values for the hypothesis test
                    variable1 = df[variable1_name].dropna().values
                    variable2 = df[variable2_name].dropna().values

                    # Perform t-test
                    t_stat, p_value = ttest_ind(variable1, variable2)
                    significance_level = 0.05

                    # Calculate mean
                    mean_variable1 = np.mean(variable1)
                    mean_variable2 = np.mean(variable2)

                    # Store results
                    null_hypothesis_results.append(f"Null hypothesis testing for columns {variable1_name} and {variable2_name}:\n")
                    null_hypothesis_results.append(f"T-Statistic: {t_stat}\n")
                    null_hypothesis_results.append(f"P-Value: {p_value}\n")
                    null_hypothesis_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")
                    null_hypothesis_results.append(f"Mean of {variable1_name}: {mean_variable1}\n")
                    null_hypothesis_results.append(f"Mean of {variable2_name}: {mean_variable2}\n")

        results.append('\n'.join(null_hypothesis_results))

        # Analyze overall test results
        total_tests = len(null_hypothesis_results) // 6
        if total_tests > 0:
            significant_tests = sum([1 for i in range(3, len(null_hypothesis_results), 6) if 'Reject' in null_hypothesis_results[i]])
            percentage_significant = (significant_tests / total_tests) * 100
            analysis = f"Overall, {significant_tests} out of {total_tests} tests ({percentage_significant:.2f}%) show significant differences between means, suggesting that there are significant differences between some of the numeric variables in the dataset."
            results.append(analysis)
        else:
            analysis = "No tests were performed as there are not enough numeric variables in the dataset."
            results.append(analysis)

        return '\n'.join(results)


    @staticmethod
    def perform_anova_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform ANOVA test
        anova_results = []
        significant_variables = set()

        for column in df.columns:
            variable_name = column

            # Extract numerical values for the ANOVA test
            variable = df[variable_name].dropna().values

            # Perform ANOVA test
            if len(variable) > 1:
                f_stat, p_value = f_oneway(variable, np.zeros_like(variable))
                anova_results.append({
                    'Variable': variable_name,
                    'F-Statistic': f_stat,
                    'P-Value': p_value,
                    'Significance': 'Reject' if p_value < significance_level else 'Fail to reject'
                })

                # Check if the test is significant
                if p_value < significance_level:
                    significant_variables.add(variable_name)

            else:
                anova_results.append({
                    'Variable': variable_name,
                    'Insufficient data': True
                })

        # Analyze overall ANOVA test results
        total_variables = len(df.columns)
        significant_variables_count = len(significant_variables)

        if total_variables > 0:
            percentage_significant = (significant_variables_count / total_variables) * 100
            analysis = f"Overall, {significant_variables_count} out of {total_variables} ANOVA tests ({percentage_significant:.2f}%) show significant differences between groups, suggesting that there are significant differences between some of the numeric variables in the dataset."
            results.append(analysis)
            if significant_variables:
                results.append("Variables with significant differences:")
                for variable in significant_variables:
                    results.append(f"- {variable}")
        else:
            analysis = "No ANOVA tests were performed as there are not enough numeric variables with sufficient data in the dataset."
            results.append(analysis)

        # Add detailed test results to the output
        detailed_results = []
        for test_result in anova_results:
            if test_result.get('Insufficient data'):
                detailed_results.append(f"ANOVA test for {test_result['Variable']}:\nInsufficient data for ANOVA test\n")
            else:
                detailed_results.append(f"ANOVA test for {test_result['Variable']}:\n")
                detailed_results.append(f"F-Statistic: {test_result['F-Statistic']}\n")
                detailed_results.append(f"P-Value: {test_result['P-Value']}\n")
                detailed_results.append(f"Significance: {test_result['Significance']}\n")
        results.extend(detailed_results)

        return '\n'.join(results)


    @staticmethod
    def perform_chi_square_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform chi-square test for each pair of columns
        chi_square_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    contingency_table = pd.crosstab(df[column1], df[column2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    chi_square_results.append(f"Chi-square test for columns {column1} and {column2}:\n")
                    chi_square_results.append(f"Chi-square statistic: {chi2}\n")
                    chi_square_results.append(f"P-value: {p_value}\n")
                    chi_square_results.append(f"Degrees of freedom: {dof}\n")
                    chi_square_results.append(f"Expected frequencies:\n{expected}\n")
                    chi_square_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")
                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall chi-square test results
        total_tests = len(chi_square_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} chi-square tests ({percentage_significant:.2f}%) show significant associations between categorical variables, suggesting that there are significant relationships between some pairs of variables in the dataset."
            results.append(analysis)
        else:
            analysis = "No chi-square tests were performed as there are not enough categorical variables with sufficient data in the dataset."
            results.append(analysis)

        results.append('\n'.join(chi_square_results))

        return '\n'.join(results)

    @staticmethod
    def perform_mann_whitney_u_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Mann-Whitney U test for each pair of columns
        mann_whitney_u_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    # Extract the non-missing and non-null values for each column
                    values1 = df[column1].dropna().values
                    values2 = df[column2].dropna().values

                    # Perform Mann-Whitney U test
                    statistic, p_value = mannwhitneyu(values1, values2)

                    mann_whitney_u_results.append(f"Mann-Whitney U test for columns {column1} and {column2}:\n")
                    mann_whitney_u_results.append(f"Mann-Whitney U statistic: {statistic}\n")
                    mann_whitney_u_results.append(f"P-value: {p_value}\n")
                    mann_whitney_u_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall Mann-Whitney U test results
        total_tests = len(mann_whitney_u_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Mann-Whitney U tests ({percentage_significant:.2f}%) show significant differences between distributions."
            results.append(analysis)
        else:
            analysis = "No Mann-Whitney U tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(mann_whitney_u_results))

        return '\n'.join(results)

    @staticmethod
    def perform_ks_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Kolmogorov-Smirnov test for each pair of columns
        ks_test_results = []
        significant_tests = []

        for column1 in df.columns:
            for column2 in df.columns:
                if column1 != column2:
                    # Perform Kolmogorov-Smirnov test
                    statistic, p_value = ks_2samp(df[column1], df[column2])

                    ks_test_results.append(f"Kolmogorov-Smirnov test for columns {column1} and {column2}:\n")
                    ks_test_results.append(f"Kolmogorov-Smirnov statistic: {statistic}\n")
                    ks_test_results.append(f"P-value: {p_value}\n")
                    ks_test_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

                    if p_value < significance_level:
                        significant_tests.append((column1, column2))

        # Analyze overall Kolmogorov-Smirnov test results
        total_tests = len(ks_test_results)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Kolmogorov-Smirnov tests ({percentage_significant:.2f}%) show significant differences between distributions."
            results.append(analysis)
        else:
            analysis = "No Kolmogorov-Smirnov tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(ks_test_results))

        return '\n'.join(results)

    @staticmethod    
    def perform_shapiro_wilk_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Initialize results
        results = []

        # Define the significance level
        significance_level = 0.05

        # Perform Shapiro-Wilk test for each column
        shapiro_test_results = []
        significant_tests = []

        for column in df.columns:
            # Perform Shapiro-Wilk test
            statistic, p_value = shapiro(df[column])

            shapiro_test_results.append(f"Shapiro-Wilk test for column {column}:\n")
            shapiro_test_results.append(f"Shapiro-Wilk statistic: {statistic}\n")
            shapiro_test_results.append(f"P-value: {p_value}\n")
            shapiro_test_results.append(f"Significance: {'Reject' if p_value < significance_level else 'Fail to reject'}\n")

            if p_value < significance_level:
                significant_tests.append(column)

        # Analyze overall Shapiro-Wilk test results
        total_tests = len(df.columns)
        significant_tests_count = len(significant_tests)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Shapiro-Wilk tests ({percentage_significant:.2f}%) show significant departure from normality."
            if significant_tests:
                analysis += "\nVariables with significant departure from normality:\n"
                for variable in significant_tests:
                    analysis += f"- {variable}\n"
            else:
                analysis += "\nNo variables show significant departure from normality."
            results.append(analysis)
        else:
            analysis = "No Shapiro-Wilk tests were performed as there are not enough data or variability in the dataset."
            results.append(analysis)

        results.append('\n'.join(shapiro_test_results))

        return '\n'.join(results)


    @staticmethod
    def perform_durbin_watson_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Create and fit the linear regression model
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions using the model
        y_pred = model.predict(X)

        # Calculate the residuals
        residuals = y - y_pred

        # Perform the Durbin-Watson test
        dw_statistic = durbin_watson(residuals)

        # Calculate critical values for a given significance level
        n = len(residuals)
        k = X.shape[1]
        critical_values = [(0, 2), (2, 1.5 + 0.15 * k), (2, 2 - 0.15 * k), (2, 4)]

        # Analyze the overall test result
        if dw_statistic < 1.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating strong positive autocorrelation."
        elif 1.5 <= dw_statistic < 2.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating no significant autocorrelation."
        elif 2.5 <= dw_statistic < 3.5:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating weak negative autocorrelation."
        else:
            result = f"The Durbin-Watson statistic is {dw_statistic}, indicating strong negative autocorrelation."

        # Add critical values and recommendation to the result
        result += "\n\nCritical Values for DW Statistic (5% Significance Level):\n"
        for i, (lower, upper) in enumerate(critical_values, start=1):
            result += f"Region {i}: ({lower}, {upper})\n"
        
        return result

    @staticmethod
    def perform_breusch_godfrey_test(file_path, max_lags=1):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Fit a linear regression model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        # Initialize lists to store results
        bg_test_statistics = []
        bg_p_values = []
        bg_dfs = []
        bg_residuals = []

        # Loop through each independent variable (column) and perform the Breusch-Godfrey test
        for column in X.columns:
            # Fit a linear regression model
            model = sm.OLS(y, sm.add_constant(X[column])).fit()

            # Perform the Breusch-Godfrey test
            bg_test_statistic, bg_p_value, bg_df, bg_residual = acorr_breusch_godfrey(model, nlags=max_lags)

            # Append the results to the lists
            bg_test_statistics.append(bg_test_statistic)
            bg_p_values.append(bg_p_value)
            bg_dfs.append(bg_df)
            bg_residuals.append(bg_residual)

        # Identify which variables have significant serial correlation in their residuals
        significant_results = [(X.columns[i], p_value < 0.05) for i, p_value in enumerate(bg_p_values)]

        # Generate an overall analysis of the test results
        overall_analysis = "Overall Analysis:\n"
        if any(significant for _, significant in significant_results):
            overall_analysis += "Variables with significant serial correlation in the residuals:\n"
            for variable, significant in significant_results:
                if significant:
                    overall_analysis += f"- {variable}\n"
        else:
            overall_analysis += "No variables have significant serial correlation in the residuals.\n"

        # Return the results for each column on separate lines along with the overall analysis
        results = overall_analysis
        for i, column in enumerate(X.columns):
            results += f"\nColumn: {column}\n"
            results += f"Breusch-Godfrey Test Statistic: {bg_test_statistics[i]}\n"
            results += f"Breusch-Godfrey P-Value: {bg_p_values[i]}\n"
            results += f"Breusch-Godfrey Degrees of Freedom: {bg_dfs[i]}\n"
            results += f"Breusch-Godfrey Residuals: {bg_residuals[i]}\n"
            if significant_results[i][1]:
                results += f"Significant serial correlation detected.\n"
            else:
                results += f"No significant serial correlation detected.\n"

        return results

    @staticmethod
    def perform_heteroscedasticity_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Initialize lists to store results
        het_test_statistics = []
        het_p_values = []
        significant_variables = []

        # Loop through each independent variable (column) and perform the Heteroscedasticity Test
        for column in X.columns:
            # Fit a linear regression model
            model = sm.OLS(y, sm.add_constant(X[column])).fit()

            # Perform the Heteroscedasticity Test
            het_test_statistic, het_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)

            # Append the results to the lists
            het_test_statistics.append(het_test_statistic)
            het_p_values.append(het_p_value)

            # Check if the test is significant
            if het_p_value < 0.05:
                significant_variables.append(column)

        # Analyze overall Heteroscedasticity Test results
        total_tests = len(het_test_statistics)
        significant_tests_count = sum(p_value < 0.05 for p_value in het_p_values)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} Heteroscedasticity tests ({percentage_significant:.2f}%) show significant results, indicating the presence of heteroscedasticity."
        else:
            analysis = "No Heteroscedasticity tests were performed as there are not enough independent variables in the dataset."

        # Add variables with significant results to the analysis
        if significant_variables:
            analysis += "\nVariables with significant heteroscedasticity:\n"
            for variable in significant_variables:
                analysis += f"- {variable}\n"
        else:
            analysis += "\nNo variables show significant heteroscedasticity."

        # Return the results for each column on separate lines along with the overall analysis
        results = analysis + "\n\n"
        for i, column in enumerate(X.columns):
            results += f"Column: {column}\n"
            results += f"Heteroscedasticity Test Statistic: {het_test_statistics[i]}\n"
            results += f"Heteroscedasticity P-Value: {het_p_values[i]}\n\n"

        return results

    @staticmethod
    def perform_multicollinearity_test(file_path):
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Convert non-numeric values to numeric
        df = Test.label_encode_non_numeric_columns(df)

        # Extract the dependent variable (Y) and independent variables (X)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        # Fit a linear regression model
        model = sm.OLS(y, sm.add_constant(X)).fit()

        # Calculate the Variance Inflation Factor for each independent variable
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
        vif["features"] = model.model.exog_names

        # Find the maximum VIF value and its corresponding feature
        max_vif = vif["VIF Factor"].max()
        max_vif_feature = vif.loc[vif["VIF Factor"].idxmax(), "features"]

        # Display the results
        multicollinearity_result = f"Multicollinearity Test\n"
        multicollinearity_result += f"Maximum VIF: {max_vif}\n"
        multicollinearity_result += f"Corresponding Feature: {max_vif_feature}\n"

        # Display multicollinear columns if present
        multicollinear_columns = vif[vif["VIF Factor"] > 10]["features"].tolist()
        if multicollinear_columns:
            multicollinearity_result += f"Multicollinear Columns: {', '.join(multicollinear_columns)}\n"

        # Analyze overall multicollinearity test results
        total_tests = len(vif)
        significant_tests_count = sum(vif["VIF Factor"] > 10)

        if total_tests > 0:
            percentage_significant = (significant_tests_count / total_tests) * 100
            analysis = f"Overall, {significant_tests_count} out of {total_tests} variables ({percentage_significant:.2f}%) exhibit multicollinearity, indicating potential issues with the model's stability and interpretability."
            multicollinearity_result += analysis
        else:
            analysis = "No multicollinearity issues detected in the dataset."
            multicollinearity_result += analysis

        return multicollinearity_result    

#Dashboard
class Dashboard:
    @staticmethod
    def label_encode_non_numeric_columns(df):
        le = LabelEncoder()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        for column in non_numeric_columns:
            # Convert values to strings before label encoding
            df[column] = df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        return df
    
    def calculate_mean(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        mean = df.to_numpy().mean()
        return mean
        
    def calculate_median(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        median = np.median(df.values)
        return median

    def calculate_mode(file_path):
     df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
     df = Dashboard.label_encode_non_numeric_columns(df)
     flattened_array = df.values.flatten()
     counts = Counter(flattened_array)
     max_count = max(counts.values())
     mode = [value for value, count in counts.items() if count == max_count]
    # If there are multiple modes, return the first one
     return mode[0] if mode else None

    def calculate_standard_deviation(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        std_deviation = df.std().mean()
        return std_deviation
    
    @staticmethod
    def recommend_charts(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        chart_types = [
            'Histogram',
            'Box plot',
            'Violin plot',
            'Dot plot',
            'Quantile-quantile plot',
            'Density plot',
            'Distribution plot',
            'Scatter plot',
            'Line plot',
            'Bar chart',
            'Pie chart',
            'Correlation matrix',
            'Heatmap',
            '3D scatter plot',
            'Parallel coordinates plot',
            'Andrews plot',
            'Chernoff faces',
            'Radar chart',
            'Treemap',
            'Sunburst chart',
            'Force-directed graph',
            'Sankey diagram',
            'Waterfall chart',
            'Covariance matrix'
        ]
        
        # Set seed for random number generator
        random.seed(42)

        recommended_charts = random.sample(chart_types, k=7)
        
        return recommended_charts
    
    @staticmethod
    def recommend_analysis(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        analysis_types = [
            'exploratory_data_analysis',
            'descriptive_statistics',
            'inferential_statistics',
            'hypothesis_testing',
            'correlation_analysis',
            'regression_analysis',
            'time_series_analysis',
            'clustering',
            'classification',
            'dimensionality_reduction'
        ]
        
        # Set seed for random number generator
        random.seed(42)

        recommended_analysis = random.sample(analysis_types, k=3)
        
        return recommended_analysis
    
    @staticmethod
    def recommend_models(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        model_types = [
            'Linear Regression',
            'Multiple Regression',
            'Logistic Regression',
            'Poisson Regression',
            'Negative Binomial Regression',
            'Ridge Regression',
            'Elastic Net Regression',
            'Time Series Regression',
            'Lasso Regression',
            'Random Forest Regression'
        ]
        
        # Set seed for random number generator
        random.seed(42)

        recommended_models = random.sample(model_types, k=3)
        
        return recommended_models

    @staticmethod
    def recommend_tests(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        tests_types = [
            'T-Tests',
            'Null and Alternative Hypothesis',
            'ANOVA',
            'Chi-Square Test',
            'Mann-Whitney U Test',
            'Kolmogorov-Smirnov Test',
            'Shapiro-Wilk Test',
            'Durbin-Watson Test',
            'Breusch-Godfrey Test',
            'Heteroscedasticity Test',
            'Multicollinearity Test'
        ]
        
        # Set seed for random number generator
        random.seed(42)

        recommended_tests = random.sample(tests_types, k=3)
        
        return recommended_tests

   

    @staticmethod
    def generate_histogram(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Label encode non-numeric columns
        df = Dashboard.label_encode_non_numeric_columns(df)
        
        # Configure Matplotlib to use a non-GUI backend
        plt.switch_backend('agg')
        plt.ioff()

        plt.figure(figsize=(6, 6))

        # Generate the histogram for the entire dataset
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].plot(kind='hist', alpha=0.5, label=column)

        plt.title('Histogram')
        plt.xlabel('Values')
        plt.legend()
        
        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        histogram_plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return histogram_plot_data
    
    @staticmethod
    def generate_line_chart(file_path):
       df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

       # Label encode non-numeric columns
       df = Dashboard.label_encode_non_numeric_columns(df)
    
       # Configure Matplotlib to use a non-GUI backend
       plt.switch_backend('agg')
       plt.ioff()

       plt.figure(figsize=(6, 6))

       # Generate the line chart for the entire dataset
       for column in df.columns:
         if pd.api.types.is_numeric_dtype(df[column]):
             df[column].plot(label=column)

       plt.title('Line Chart')
       plt.xlabel('Values')
       plt.ylabel('Counts')
       plt.legend()  # Add legend to the plot

        # Save the plot to a BytesIO object
       image_stream = BytesIO()
       plt.savefig(image_stream, format='png')
       image_stream.seek(0)
       line_chart_data = base64.b64encode(image_stream.read()).decode('utf-8')
       plt.close()

       return line_chart_data

    @staticmethod
    def generate_heatmap(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Label encode non-numeric columns
        df = Dashboard.label_encode_non_numeric_columns(df)
        
        # Configure Matplotlib to use a non-GUI backend
        plt.switch_backend('agg')
        plt.ioff()

        plt.figure(figsize=(9, 9))

        # Generate heatmap using seaborn
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")

        plt.title('Heatmap of Dataset Correlation')
        
        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        heatmap_plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return heatmap_plot_data
    
    @staticmethod
    def generate_pie_chart(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Label encode non-numeric columns
        df = Dashboard.label_encode_non_numeric_columns(df)

        df = df.clip(lower=0)
    
        # Configure Matplotlib to use a non-GUI backend
        plt.switch_backend('agg')
        plt.ioff()

        plt.figure(figsize=(5, 5))
        

        # Generate the pie chart for the entire dataset
        sizes = df.sum()
        labels = sizes.index  # Use column names as labels
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

        plt.title('Pie Chart')
        plt.legend()

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        pie_chart_data = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        return pie_chart_data

    @staticmethod
    def generate_eda(file_path):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = Dashboard.label_encode_non_numeric_columns(df)
        
        eda_result = f"Exploratory Data Analysis for {file_path}:\n"

        # Basic information about the dataset
        eda_result += f"Number of rows: {len(df)}\n"
        eda_result += f"Number of columns: {len(df.columns)}\n"
        eda_result += f"All Variables: {', '.join(df.columns)}\n"
        eda_result += "\n"

        # Summary statistics for all numerical columns
        eda_result += "Summary Statistics for All Numerical Columns:\n"
        eda_result += df.describe().to_string() + "\n"
        eda_result += "\n"

        # Info about data types and missing values for all variables
        eda_result += "Data Types and Missing Values for All Variables:\n"
        info_str = str(df.info())
        eda_result += info_str if info_str else "No missing values\n"
        eda_result += "\n"

        # Univariate Analysis - Count of unique values for each column
        eda_result += "Count of Unique Values for Each Column:\n"
        eda_result += df.nunique().to_string() + "\n"
        eda_result += "\n"

        # Correlation matrix for all numerical columns
        eda_result += "Correlation Matrix for All Numerical Columns:\n"
        eda_result += df.corr().to_string() + "\n"

        return eda_result

class DataFile:
    # Set the uploads directory and allowed file extensions
    UPLOADS_DIR = 'uploads'  # Set your actual uploads directory
    ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @staticmethod
    def upload_and_convert(file):
        try:
            # Check if the file has a valid extension
            if not DataFile.allowed_file(file.filename):
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

            # Save the uploaded file
            upload_path = os.path.join(DataFile.UPLOADS_DIR, file.filename)
            file.save(upload_path)

            # Read the CSV or Excel file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(upload_path)
            elif file.filename.endswith(('.xls', '.xlsx')):
                # Specify the engine as 'openpyxl' for reading Excel files
                df = pd.read_excel(upload_path, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

            # Convert non-numerical values to numerical values
            df = df.apply(pd.to_numeric, errors='coerce')

            # Save the modified DataFrame to a new CSV file
            output_file_path = os.path.join(DataFile.UPLOADS_DIR, file.filename.replace('.', '_numerical.'))
            df.to_csv(output_file_path, index=False)

            return output_file_path

        except Exception as e:
            return f"An error occurred: {str(e)}"

    @staticmethod
    # Function to get attributes from the dataset (column names)
    def get_attributes_from_dataset(dataset_path):
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(dataset_path, engine='openpyxl')
        else:
            return []

        return list(df.columns)
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl') 
    
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

# Define route to 'index.html' template
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/ourteam')
def ourteam():
    return render_template('ourteam.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Define route to 'solutions.html' template
@app.route('/solutions')
def solutions():
    # Check if user is logged in
    if 'username' in session:
        return render_template('solutions.html')
    else:
        # If user is not logged in, redirect to login page
        return redirect(url_for('login'))

# Define route to 'chartsmetrices.html' template
@app.route('/chartsmetrices')
def chartsmetrices():
    return render_template('chartsmetrices.html')

# Define route to 'dataanlysis.html' template
@app.route('/analysisdescription')
def analysisdescription():
    return render_template('analysisdescription.html')

# Define route to 'regressionmodels.html' template
@app.route('/regressionmodelsdescription')
def regressionmodelsdescription():
    return render_template('regressionmodelsdescription.html')

# Define route to 'test.html' template
@app.route('/testsdescription')
def testsdescription():
    return render_template('testsdescription.html')


# Define route to 'askwhiz.html' page
@app.route('/askwhiz', methods=['GET', 'POST'])
def askwhiz():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('askwhiz.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('askwhiz.html', error_message='No selected file')

        if file and DataFile.allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)

            # Detect file format and read accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return render_template('askwhiz.html', error_message='Unsupported file format')

            # Generate HTML from DataFrame and render template
            file_content_html = df.to_html(classes="table table-striped")
            return render_template('askwhiz.html', file_uploaded=True, filename=file.filename, file_content_html=file_content_html)

    return render_template('askwhiz.html')



# Define route to 'chatbot' function
@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
    chart_data = {}
    model_data = {}
    test_results = {}
    analysis_results = {}

    file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))

    # Detect file format and read accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path) 
    
    file_content_html = df.to_html(classes="table table-striped")

    chart_types = request.form.getlist('chart-type')
    analysis_types = request.form.getlist('analysis-type')
    model_types = request.form.getlist('model-type')
    test_types = request.form.getlist('test-type')
    queryInput = request.form.get('queryInput')
    query = request.form.get('chart-query')

    chart_data = {chart_type: AskWhizVisualization.generate_chart(file_path, chart_type) for chart_type in chart_types}

    for analysis_type in analysis_types:
        if analysis_type == 'exploratory_data_analysis':
            analysis_results[analysis_type] = AskWhizAnalysis.eda(file_path)
        elif analysis_type == 'descriptive_statistics':
            analysis_results[analysis_type] = AskWhizAnalysis.descriptive_statistics(file_path)
        elif analysis_type == 'inferential_statistics':
            analysis_results[analysis_type] = AskWhizAnalysis.inferential_statistics(file_path)
        elif analysis_type == 'hypothesis_testing':
            analysis_results[analysis_type] = AskWhizAnalysis.hypothesis_testing(file_path)
        elif analysis_type == 'correlation_analysis':
            analysis_results[analysis_type] = AskWhizAnalysis.correlation_analysis(file_path)
        elif analysis_type == 'regression_analysis':
            analysis_results[analysis_type] = AskWhizAnalysis.regression_analysis(file_path)
        elif analysis_type == 'time_series_analysis':
            analysis_results[analysis_type] = AskWhizAnalysis.time_series_analysis(file_path)
        elif analysis_type == 'clustering':
            analysis_results[analysis_type] = AskWhizAnalysis.clustering(file_path)
        elif analysis_type == 'classification':
            analysis_results[analysis_type] = AskWhizAnalysis.classification(file_path)
        elif analysis_type == 'dimensionality_reduction':
            analysis_results[analysis_type] = AskWhizAnalysis.dimensionality_reduction(file_path)
        else:
            analysis_results[analysis_type] = "Invalid analysis type"

    model_data = {model_type: AskWhizModel.apply_model(file_path, model_type) for model_type in model_types}

    for test_type in test_types:
        if test_type == 't_test':
            test_results[test_type] = AskWhizTest.t_test(file_path)
        elif test_type == 'null_and_hypothesis_test':
            test_results[test_type] = AskWhizTest.null_and_hypothesis_test(file_path)
        elif test_type == 'anova':
            test_results[test_type] = AskWhizTest.anova_test(file_path)
        elif test_type == 'chi_square_test':
            test_results[test_type] = AskWhizTest.chi_square_test(file_path)
        elif test_type == 'mann_whitney_u_test':
            test_results[test_type] = AskWhizTest.mann_whitney_u_test(file_path)
        elif test_type == 'kolmogorov_smirnov_test':
            test_results[test_type] = AskWhizTest.ks_test(file_path)
        elif test_type == 'shapiro_wilk_test':
            test_results[test_type] = AskWhizTest.shapiro_wilk_test(file_path)
        elif test_type == 'durbin_watson_test':
            test_results[test_type] = AskWhizTest.durbin_watson_test(file_path)
        elif test_type == 'breusch_godfrey_test':
            test_results[test_type] = AskWhizTest.breusch_godfrey_test(file_path, max_lags=1)
        elif test_type == 'heteroscedasticity_test':
            test_results[test_type] = AskWhizTest.heteroscedasticity_test(file_path)
        elif test_type == 'multicollinearity_test':
            test_results[test_type] = AskWhizTest.multicollinearity_test(file_path)
        else:
            test_results[test_type] = "Invalid test type"

    # Initialize the AI model
    ai = AI(api_key='# Your Secret Key')

    # Read dataset
    dataset = AI.read_dataset(file_path)

    # Construct prompt
    prompt = (
        "Please provide only shortest possible, very relevant and highly precise answer to the user query. You must provide the exact insights/trends/analytics of overall dataset."
        f"Do not include any additional or unneccessary information.\n\n"
        f"Dataset:\n{dataset}\n\n"
        f"User Query: {queryInput}\n\n"
        "Very Short and Precise Answer:"
    )

    # Generate response
    response = ai.generate_text(prompt=prompt)


    return render_template('askwhiz.html', file_uploaded=True, filename=request.form.get('filename'), query=query, response=response, chart_data=chart_data, 
                           analysis_results=analysis_results, model_data=model_data, test_results=test_results, file_content_html=file_content_html)



# Define route to 'dashboard.html' template
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    datasets=[]
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('dashboard.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('dashboard.html', error_message='No selected file')

        if file and DataFile.allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)

            # Calculate the statistics of the dataset
            mean = Dashboard.calculate_mean(file_path)
            median = Dashboard.calculate_median(file_path)
            mode = Dashboard.calculate_mode(file_path)
            std_deviation = Dashboard.calculate_standard_deviation(file_path)
            recommend_charts = Dashboard.recommend_charts(file_path)
            recommend_analysis = Dashboard.recommend_analysis(file_path)
            recommend_models = Dashboard.recommend_models(file_path)
            recommend_tests = Dashboard.recommend_tests(file_path)
            histogram = Dashboard.generate_histogram(file_path)
            linechart = Dashboard.generate_line_chart(file_path)
            piechart = Dashboard.generate_pie_chart(file_path)
            heatmap = Dashboard.generate_heatmap(file_path)
            exploratoryanalysis = Dashboard.generate_eda(file_path)
            
            return render_template('dashboard.html', file_uploaded=True, filename=file.filename, mean=mean, median=median, mode=mode, std_deviation=std_deviation, recommend_charts=recommend_charts, recommend_analysis=recommend_analysis, recommend_models=recommend_models, recommend_tests=recommend_tests,histogram=histogram, linechart=linechart, heatmap=heatmap, piechart=piechart, exploratoryanalysis=exploratoryanalysis)
    # List datasets regardless of the request method
    datasets = os.listdir(UPLOADS_DIR)

    return render_template('dashboard.html', datasets=datasets)

# Define route to 'visualization.html' template
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    datasets = []  # Initialize datasets list

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('visualization.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('visualization.html', error_message='No selected file')

        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)

            return redirect(url_for('chart_types', filename=file.filename))

    # List datasets regardless of the request method
    datasets = os.listdir(UPLOADS_DIR)

    return render_template('visualization.html', datasets=datasets)

# Define route to 'chartTypes.html' template
@app.route('/chart_types/<filename>', methods=['GET'])
def chart_types(filename):
    file_path = os.path.join(UPLOADS_DIR, filename)
    attributes = DataFile.get_attributes_from_dataset(file_path)
    return render_template('chartTypes.html', filename=filename, attributes=attributes)

# Define route to 'generate_chart' function in visualization
@app.route('/generate_chart', methods=['POST'])
def generate_chart_route():
    file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))
    chart_type = request.form.get('chart-type')
    x_label = request.form.get('x-label')
    y_label = request.form.get('y-label')
    z_label = request.form.get('z-label')

    chart = Visualization.generate_chart(file_path, chart_type, x_label, y_label, z_label)
    attributes = DataFile.get_attributes_from_dataset(file_path)

    session['chart'] = chart
    
    return render_template('chartTypes.html', file_uploaded=True, filename=request.form.get('filename'), attributes=attributes, chart=chart, chart_type=chart_type)

# Define route to 'publishChart' function in visualization
@app.route('/publishChart')
def publishChart():
    chart_data = session.get('chart_data', None)
    return render_template('publishChart.html', chart_data=chart_data)


# Define route to 'analysis.html' template
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    datasets = []  # Initialize datasets list

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('analysis.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('analysis.html', error_message='No selected file')

        if file and DataFile.allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)
            attributes = DataFile.get_attributes_from_dataset(file_path)

            return render_template('analysis.html', file_uploaded=True, filename=file.filename, attributes=attributes)
        
    # List datasets regardless of the request method
    datasets = os.listdir(UPLOADS_DIR)

    return render_template('analysis.html', datasets=datasets)

# Define route to 'analyze' function in analysis
@app.route('/analyze', methods=['POST'])
def perform_analysis():
    file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))
    analysis_type = request.form.get('analysis-type')

    if analysis_type == 'exploratory_data_analysis':
        analysis = Analysis.perform_eda(file_path)
    elif analysis_type == 'descriptive_statistics':
        analysis = Analysis.perform_descriptive_statistics(file_path)
    elif analysis_type == 'inferential_statistics':
        analysis = Analysis.perform_inferential_statistics(file_path)
    elif analysis_type == 'hypothesis_testing':
        analysis = Analysis.perform_hypothesis_testing(file_path)
    elif analysis_type == 'regression_analysis':
        analysis = Analysis.perform_regression_analysis(file_path)
    elif analysis_type == 'correlation_analysis':
        analysis = Analysis.perform_correlation_analysis(file_path)
    elif analysis_type == 'time_series_analysis':
        analysis = Analysis.perform_time_series_analysis(file_path)   
    elif analysis_type == 'clustering':
        analysis = Analysis.perform_clustering(file_path) 
    elif analysis_type == 'classification':
        analysis = Analysis.perform_classification(file_path)
    elif analysis_type == 'dimensionality_reduction':
        analysis = Analysis.perform_dimensionality_reduction(file_path)
    
    else:
        analysis = "Invalid analysis type"

    return render_template('analysis.html', file_uploaded=True, filename=request.form.get('filename'), analysis=analysis)


# Define route to 'model.html' template
@app.route('/model', methods=['GET', 'POST'])
def model():
    datasets = []  # Initialize datasets list
    if request.method == 'POST':
        if 'file' not in request.files:
           return render_template('model.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('model.html', error_message='No selected file')

        if file and DataFile.allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)
            attributes = DataFile.get_attributes_from_dataset(file_path)

            return render_template('model.html', file_uploaded=True, filename=file.filename, attributes=attributes)
    # List datasets regardless of the request method
    datasets = os.listdir(UPLOADS_DIR)
    return render_template('model.html', datasets=datasets)


# Define the root route for rendering the 'applymodel' function in regression models
@app.route('/apply_model', methods=['POST'])
@staticmethod
def apply_model_route():
    file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))

    # Get user inputs
    model_type = request.form.get('model-type')
    independent_variable = request.form.get('Independent_variable')
    independent_variable2 = request.form.get('Independent_variable2')
    dependent_variable = request.form.get('Dependent_variable')

    # Apply the model
    model = RegressionModel.apply_model(file_path, model_type, independent_variable, independent_variable2, dependent_variable)
    attributes = DataFile.get_attributes_from_dataset(file_path)

    return render_template('model.html', file_uploaded=True, filename=request.form.get('filename'), attributes=attributes, model=model)

# Define route to 'test.html' template
@app.route('/test', methods=['GET', 'POST'])
def test():
    datasets = []  
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('test.html', error_message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('test.html', error_message='No selected file')

        if file and DataFile.allowed_file(file.filename):
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)
            attributes = DataFile.get_attributes_from_dataset(file_path)

            return render_template('test.html', file_uploaded=True, filename=file.filename, attributes=attributes)
    # List datasets regardless of the request method
    datasets = os.listdir(UPLOADS_DIR)
    return render_template('test.html', datasets=datasets)

# Define route to 'perform_test' function in test
@app.route('/perform_test', methods=['POST'])
def perform_test():
    file_path = os.path.join(UPLOADS_DIR, request.form.get('filename'))
    test_type = request.form.get('test-type')

    if test_type == 't_test':
        test = Test.perform_t_test(file_path)
    elif test_type == 'null_and_hypothesis_test':
        test = Test.perform_null_and_hypothesis_test(file_path)
    elif test_type == 'anova':
        test = Test.perform_anova_test(file_path)
    elif test_type == 'chi_square_test':
        test = Test.perform_chi_square_test(file_path)
    elif test_type == 'mann_whitney_u_test':
        test = Test.perform_mann_whitney_u_test(file_path)
    elif test_type == 'kolmogorov_smirnov_test':
        test = Test.perform_ks_test(file_path)
    elif test_type == 'shapiro_wilk_test':
        test = Test.perform_shapiro_wilk_test(file_path)
    elif test_type == 'durbin_watson_test':
        test = Test.perform_durbin_watson_test(file_path)
    elif test_type == 'breusch_godfrey_test':
        test = Test.perform_breusch_godfrey_test(file_path)
    elif test_type == 'heteroscedasticity_test':
        test = Test.perform_heteroscedasticity_test(file_path)
    elif test_type == 'multicollinearity_test':
        test = Test.perform_multicollinearity_test(file_path)   
    else:
        test = "Invalid test type"

    return render_template('test.html', file_uploaded=True, filename=request.form.get('filename'), test=test)
