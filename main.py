from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import json
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import secrets
import hashlib

app = FastAPI()

# Security: HTTP Basic Authentication
security = HTTPBasic()

# Security: Simple authentication (use environment variables in production)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = hashlib.sha256(os.environ.get("ADMIN_PASSWORD", "ushika2024").encode()).hexdigest()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """Simple authentication check"""
    username_correct = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    password_correct = secrets.compare_digest(
        hashlib.sha256(credentials.password.encode()).hexdigest(), 
        ADMIN_PASSWORD_HASH
    )
    
    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, user: str = Depends(authenticate)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save-data")
async def save_data(request: Request, user: str = Depends(authenticate)):
    data = await request.json()
    return {"status": "success", "data": data}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), user: str = Depends(authenticate)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        contents = await file.read()
        
        # Security: Limit file size to prevent abuse
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Security: Limit number of rows to prevent memory exhaustion
        if len(df) > 10000:
            raise HTTPException(status_code=413, detail="Too many rows. Maximum is 10,000 rows.")
        
        # Security: Limit number of columns
        if len(df.columns) > 50:
            raise HTTPException(status_code=413, detail="Too many columns. Maximum is 50 columns.")
        
        # Clean the data to handle large numbers with commas and missing values
        def clean_cell_value(value):
            """Clean individual cell values for JSON compatibility"""
            if pd.isna(value) or value == '' or str(value).strip() == '':
                return None
            
            # Convert to string first
            str_value = str(value).strip()
            
            # Try to clean numeric values with commas
            if ',' in str_value and str_value.replace(',', '').replace('.', '').replace('-', '').isdigit():
                try:
                    # Remove commas and convert to float, then to int if it's a whole number
                    clean_num = float(str_value.replace(',', ''))
                    if clean_num.is_integer():
                        return int(clean_num)
                    return clean_num
                except (ValueError, OverflowError):
                    return str_value
            
            # Try to convert to numeric if possible
            try:
                num_value = float(str_value)
                # Check if it's JSON-safe (not inf or nan)
                if not (np.isinf(num_value) or np.isnan(num_value)):
                    if num_value.is_integer():
                        return int(num_value)
                    return num_value
                else:
                    return str_value
            except (ValueError, OverflowError):
                return str_value
        
        # Apply cleaning to all data
        cleaned_data = []
        for row in df.values:
            cleaned_row = [clean_cell_value(cell) for cell in row]
            cleaned_data.append(cleaned_row)
        
        table_data = {
            "columns": df.columns.tolist(),
            "data": cleaned_data,
            "filename": file.filename
        }
        
        return JSONResponse(content=table_data)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/generate-scatter-plot")
async def generate_scatter_plot(request: Request, user: str = Depends(authenticate)):
    try:
        data = await request.json()
        x_column_idx = int(data.get('x_column'))
        y_column_idx = int(data.get('y_column'))
        hover_column_idx = data.get('hover_column')
        color_column_idx = data.get('color_column')
        filter_column_idx = data.get('filter_column')
        filter_value = data.get('filter_value')
        filter2_column_idx = data.get('filter2_column')
        filter2_value = data.get('filter2_value')
        ols_enabled = data.get('ols_enabled', False)
        csv_data = data.get('csv_data')
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No CSV data provided")
        
        # Apply first filtering if specified
        filtered_data = csv_data['data']
        filter_info = ""
        
        if filter_column_idx is not None and filter_column_idx != "" and filter_value is not None and filter_value != "":
            filter_column_idx = int(filter_column_idx)
            filtered_data = [row for row in filtered_data if str(row[filter_column_idx]) == str(filter_value)]
            filter_info += f"{csv_data['columns'][filter_column_idx]} = {filter_value}"
            
            if len(filtered_data) == 0:
                raise HTTPException(status_code=400, detail=f"No data found for first filter: {csv_data['columns'][filter_column_idx]} = {filter_value}")
        
        # Apply second filtering if specified
        if filter2_column_idx is not None and filter2_column_idx != "" and filter2_value is not None and filter2_value != "":
            filter2_column_idx = int(filter2_column_idx)
            filtered_data = [row for row in filtered_data if str(row[filter2_column_idx]) == str(filter2_value)]
            
            if filter_info:
                filter_info += f" & {csv_data['columns'][filter2_column_idx]} = {filter2_value}"
            else:
                filter_info = f"{csv_data['columns'][filter2_column_idx]} = {filter2_value}"
            
            if len(filtered_data) == 0:
                raise HTTPException(status_code=400, detail=f"No data found after applying both filters")
        
        # Extract data for plotting
        x_values = [row[x_column_idx] for row in filtered_data if row[x_column_idx] is not None]
        y_values = [row[y_column_idx] for row in filtered_data if row[y_column_idx] is not None]
        
        # Extract hover column data if specified
        hover_values = []
        if hover_column_idx is not None and hover_column_idx != "":
            hover_column_idx = int(hover_column_idx)
            hover_values = [row[hover_column_idx] for row in filtered_data]
        
        # Extract color column data if specified
        color_values = []
        if color_column_idx is not None and color_column_idx != "":
            color_column_idx = int(color_column_idx)
            color_values = [row[color_column_idx] for row in filtered_data]
        
        # Convert to numeric, filter out non-numeric values
        x_numeric = []
        y_numeric = []
        hover_text = []
        color_text = []
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            try:
                x_num = float(x)
                y_num = float(y)
                x_numeric.append(x_num)
                y_numeric.append(y_num)
                
                # Add hover text if hover column is specified
                if hover_values and i < len(hover_values):
                    hover_text.append(str(hover_values[i]))
                else:
                    hover_text.append("")
                
                # Add color text if color column is specified
                if color_values and i < len(color_values):
                    color_text.append(str(color_values[i]))
                else:
                    color_text.append("")
            except (ValueError, TypeError):
                continue
        
        if len(x_numeric) == 0 or len(y_numeric) == 0:
            raise HTTPException(status_code=400, detail="Selected columns must contain numeric data")
        
        # Create hover template - show only selected hover column data if specified
        if hover_column_idx is not None and hover_column_idx != "":
            hover_template = f"<b>{csv_data['columns'][hover_column_idx]}</b><br>%{{text}}<extra></extra>"
        else:
            # If no hover column selected, show default X and Y values
            hover_template = f"<b>Data Point</b><br>" + \
                            f"{csv_data['columns'][x_column_idx]}: %{{x}}<br>" + \
                            f"{csv_data['columns'][y_column_idx]}: %{{y}}<br>" + \
                            "<extra></extra>"
        
        # Update title to include filter information
        plot_title = f"Interactive Scatter Plot: {csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]}"
        if filter_info:
            plot_title += f" (Filtered: {filter_info})"
        
        # Create Plotly scatter plot with color coding
        if color_column_idx is not None and color_column_idx != "" and color_text:
            # Get unique values for color mapping
            unique_colors = list(set(color_text))
            color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
            ]
            
            # Create color mapping
            color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_colors)}
            colors = [color_map[val] for val in color_text]
            
            scatter_data = go.Scatter(
                x=x_numeric,
                y=y_numeric,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=1, color='rgba(0, 0, 0, 0.3)')
                ),
                name=f"{csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]}",
                hovertemplate=hover_template,
                text=color_text,
                customdata=color_text
            )
        else:
            # Default single color
            scatter_data = go.Scatter(
                x=x_numeric,
                y=y_numeric,
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(54, 162, 235, 0.7)',
                    line=dict(width=1, color='rgba(54, 162, 235, 1)')
                ),
                name=f"{csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]}",
                hovertemplate=hover_template
            )
        
        # Add hover text if hover column is specified
        if hover_column_idx is not None and hover_column_idx != "" and hover_text:
            scatter_data.text = hover_text
        
        fig = go.Figure(data=scatter_data)
        
        # Add OLS trend line if enabled
        if ols_enabled and len(x_numeric) >= 2:
            try:
                # Prepare data for linear regression
                X = np.array(x_numeric).reshape(-1, 1)
                y = np.array(y_numeric)
                
                # Check for invalid values
                if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)) or np.any(np.isinf(y)):
                    raise ValueError("Data contains NaN or infinite values")
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate trend line points
                x_trend = np.linspace(min(x_numeric), max(x_numeric), 100)
                y_trend = model.predict(x_trend.reshape(-1, 1))
                
                # Calculate R-squared
                r_squared = model.score(X, y)
                
                # Check for invalid trend line values
                if np.any(np.isnan(y_trend)) or np.any(np.isinf(y_trend)):
                    raise ValueError("Trend line contains invalid values")
                
                # Add trend line to plot
                fig.add_trace(go.Scatter(
                    x=x_trend.tolist(),
                    y=y_trend.tolist(),
                    mode='lines',
                    name=f'OLS Trend Line (R² = {r_squared:.3f})',
                    line=dict(
                        color='red',
                        width=2,
                        dash='dash'
                    ),
                    hovertemplate=f'<b>OLS Trend Line</b><br>' +
                                f'R² = {r_squared:.3f}<br>' +
                                f'Slope = {model.coef_[0]:.3f}<br>' +
                                f'Intercept = {model.intercept_:.3f}<br>' +
                                '<extra></extra>'
                ))
            except Exception as e:
                # If OLS calculation fails, continue without trend line
                pass
        
        # Add color legend if color coding is used
        if color_column_idx is not None and color_column_idx != "" and color_text:
            # Create dummy traces for legend
            for i, unique_val in enumerate(unique_colors):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color_palette[i % len(color_palette)]),
                    name=str(unique_val),
                    showlegend=True,
                    legendgroup=str(unique_val)
                ))
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title=csv_data['columns'][x_column_idx],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title=csv_data['columns'][y_column_idx],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            hovermode='closest',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Convert to JSON for frontend
        graph_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return JSONResponse(content={"plot": graph_json})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating scatter plot: {str(e)}")

@app.post("/generate-3d-scatter-plot")
async def generate_3d_scatter_plot(request: Request, user: str = Depends(authenticate)):
    try:
        data = await request.json()
        x_column_idx = int(data.get('x_column'))
        y_column_idx = int(data.get('y_column'))
        z_column_idx = int(data.get('z_column'))
        hover_column_idx = data.get('hover_column')
        color_column_idx = data.get('color_column')
        filter_column_idx = data.get('filter_column')
        filter_value = data.get('filter_value')
        filter2_column_idx = data.get('filter2_column')
        filter2_value = data.get('filter2_value')
        ols_enabled = data.get('ols_enabled', False)
        csv_data = data.get('csv_data')
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No CSV data provided")
        
        # Apply first filtering if specified
        filtered_data = csv_data['data']
        filter_info = ""
        
        if filter_column_idx is not None and filter_column_idx != "" and filter_value is not None and filter_value != "":
            filter_column_idx = int(filter_column_idx)
            filtered_data = [row for row in filtered_data if str(row[filter_column_idx]) == str(filter_value)]
            filter_info += f"{csv_data['columns'][filter_column_idx]} = {filter_value}"
            
            if len(filtered_data) == 0:
                raise HTTPException(status_code=400, detail=f"No data found for first filter: {csv_data['columns'][filter_column_idx]} = {filter_value}")
        
        # Apply second filtering if specified
        if filter2_column_idx is not None and filter2_column_idx != "" and filter2_value is not None and filter2_value != "":
            filter2_column_idx = int(filter2_column_idx)
            filtered_data = [row for row in filtered_data if str(row[filter2_column_idx]) == str(filter2_value)]
            
            if filter_info:
                filter_info += f" & {csv_data['columns'][filter2_column_idx]} = {filter2_value}"
            else:
                filter_info = f"{csv_data['columns'][filter2_column_idx]} = {filter2_value}"
            
            if len(filtered_data) == 0:
                raise HTTPException(status_code=400, detail=f"No data found after applying both filters")
        
        # Extract data for plotting
        x_values = [row[x_column_idx] for row in filtered_data if row[x_column_idx] is not None]
        y_values = [row[y_column_idx] for row in filtered_data if row[y_column_idx] is not None]
        z_values = [row[z_column_idx] for row in filtered_data if row[z_column_idx] is not None]
        
        # Extract hover column data if specified
        hover_values = []
        if hover_column_idx is not None and hover_column_idx != "":
            hover_column_idx = int(hover_column_idx)
            hover_values = [row[hover_column_idx] for row in filtered_data]
        
        # Extract color column data if specified
        color_values = []
        if color_column_idx is not None and color_column_idx != "":
            color_column_idx = int(color_column_idx)
            color_values = [row[color_column_idx] for row in filtered_data]
        
        # Convert to numeric, filter out non-numeric values
        x_numeric = []
        y_numeric = []
        z_numeric = []
        hover_text = []
        color_text = []
        for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
            try:
                x_num = float(x)
                y_num = float(y)
                z_num = float(z)
                x_numeric.append(x_num)
                y_numeric.append(y_num)
                z_numeric.append(z_num)
                
                # Add hover text if hover column is specified
                if hover_values and i < len(hover_values):
                    hover_text.append(str(hover_values[i]))
                else:
                    hover_text.append("")
                
                # Add color text if color column is specified
                if color_values and i < len(color_values):
                    color_text.append(str(color_values[i]))
                else:
                    color_text.append("")
            except (ValueError, TypeError):
                continue
        
        if len(x_numeric) == 0 or len(y_numeric) == 0 or len(z_numeric) == 0:
            raise HTTPException(status_code=400, detail="Selected columns must contain numeric data")
        
        # Create hover template - show only selected hover column data if specified
        if hover_column_idx is not None and hover_column_idx != "":
            hover_template = f"<b>{csv_data['columns'][hover_column_idx]}</b><br>%{{text}}<extra></extra>"
        else:
            # If no hover column selected, show default X, Y, and Z values
            hover_template = f"<b>Data Point</b><br>" + \
                            f"{csv_data['columns'][x_column_idx]}: %{{x}}<br>" + \
                            f"{csv_data['columns'][y_column_idx]}: %{{y}}<br>" + \
                            f"{csv_data['columns'][z_column_idx]}: %{{z}}<br>" + \
                            "<extra></extra>"
        
        # Update title to include filter information
        plot_title = f"Interactive 3D Scatter Plot: {csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]} vs {csv_data['columns'][z_column_idx]}"
        if filter_info:
            plot_title += f" (Filtered: {filter_info})"
        
        # Create Plotly 3D scatter plot with color coding
        if color_column_idx is not None and color_column_idx != "" and color_text:
            # Get unique values for color mapping
            unique_colors = list(set(color_text))
            color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
            ]
            
            # Create color mapping
            color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_colors)}
            colors = [color_map[val] for val in color_text]
            
            scatter_data = go.Scatter3d(
                x=x_numeric,
                y=y_numeric,
                z=z_numeric,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=1, color='rgba(0, 0, 0, 0.3)')
                ),
                name=f"{csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]} vs {csv_data['columns'][z_column_idx]}",
                hovertemplate=hover_template,
                text=color_text,
                customdata=color_text
            )
        else:
            # Default single color
            scatter_data = go.Scatter3d(
                x=x_numeric,
                y=y_numeric,
                z=z_numeric,
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(54, 162, 235, 0.7)',
                    line=dict(width=1, color='rgba(54, 162, 235, 1)')
                ),
                name=f"{csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]} vs {csv_data['columns'][z_column_idx]}",
                hovertemplate=hover_template
            )
        
        # Add hover text if hover column is specified
        if hover_column_idx is not None and hover_column_idx != "" and hover_text:
            scatter_data.text = hover_text
        
        fig = go.Figure(data=scatter_data)
        
        # Add OLS trend line for 3D plots (using parametric curve fitting)
        if ols_enabled and len(x_numeric) >= 3:
            try:
                # Convert to numpy arrays
                x_array = np.array(x_numeric)
                y_array = np.array(y_numeric)
                z_array = np.array(z_numeric)
                
                # Check for invalid values
                if np.any(np.isnan(x_array)) or np.any(np.isnan(y_array)) or np.any(np.isnan(z_array)) or \
                   np.any(np.isinf(x_array)) or np.any(np.isinf(y_array)) or np.any(np.isinf(z_array)):
                    raise ValueError("3D Data contains NaN or infinite values")
                
                # Sort points by their distance from the centroid to create a smooth line
                centroid_x, centroid_y, centroid_z = np.mean(x_array), np.mean(y_array), np.mean(z_array)
                
                # Calculate distances from centroid and sort
                distances = np.sqrt((x_array - centroid_x)**2 + (y_array - centroid_y)**2 + (z_array - centroid_z)**2)
                sorted_indices = np.argsort(distances)
                
                # Create sorted arrays
                x_sorted = x_array[sorted_indices]
                y_sorted = y_array[sorted_indices]
                z_sorted = z_array[sorted_indices]
                
                # Create parametric variable (0 to 1)
                t = np.linspace(0, 1, len(x_sorted))
                
                # Fit polynomial curves for each coordinate as function of parameter t
                try:
                    # Use degree 2 polynomial for smooth curves, but fall back to degree 1 if needed
                    degree = min(2, len(x_sorted) - 1)
                    
                    x_coeffs = np.polyfit(t, x_sorted, degree)
                    y_coeffs = np.polyfit(t, y_sorted, degree)
                    z_coeffs = np.polyfit(t, z_sorted, degree)
                    
                    # Generate smooth trend line with more points
                    t_smooth = np.linspace(0, 1, 50)
                    x_trend = np.polyval(x_coeffs, t_smooth)
                    y_trend = np.polyval(y_coeffs, t_smooth)
                    z_trend = np.polyval(z_coeffs, t_smooth)
                    
                    # Calculate R-squared for the trend (using Z as dependent variable)
                    # For 3D, we'll calculate a composite R-squared
                    z_predicted = np.polyval(z_coeffs, t)
                    ss_res = np.sum((z_sorted - z_predicted) ** 2)
                    ss_tot = np.sum((z_sorted - np.mean(z_sorted)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Check for invalid trend line values
                    if np.any(np.isnan(x_trend)) or np.any(np.isnan(y_trend)) or np.any(np.isnan(z_trend)) or \
                       np.any(np.isinf(x_trend)) or np.any(np.isinf(y_trend)) or np.any(np.isinf(z_trend)):
                        raise ValueError("3D Trend line contains invalid values")
                    
                    # Add 3D trend line to plot
                    fig.add_trace(go.Scatter3d(
                        x=x_trend.tolist(),
                        y=y_trend.tolist(),
                        z=z_trend.tolist(),
                        mode='lines',
                        name=f'3D Trend Line (R² = {r_squared:.3f})',
                        line=dict(
                            color='red',
                            width=6
                        ),
                        hovertemplate=f'<b>3D Trend Line</b><br>' +
                                    f'R² = {r_squared:.3f}<br>' +
                                    f'X: %{{x:.2f}}<br>' +
                                    f'Y: %{{y:.2f}}<br>' +
                                    f'Z: %{{z:.2f}}<br>' +
                                    '<extra></extra>'
                    ))
                    
                except np.linalg.LinAlgError:
                    # Fall back to simple linear interpolation if polynomial fitting fails
                    # Simple linear trend through the data
                    n_points = min(20, len(x_sorted))
                    indices = np.linspace(0, len(x_sorted)-1, n_points).astype(int)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_sorted[indices].tolist(),
                        y=y_sorted[indices].tolist(), 
                        z=z_sorted[indices].tolist(),
                        mode='lines',
                        name='3D Trend Line (Linear)',
                        line=dict(
                            color='red',
                            width=6
                        ),
                        hovertemplate='<b>3D Trend Line</b><br>' +
                                    f'X: %{{x:.2f}}<br>' +
                                    f'Y: %{{y:.2f}}<br>' +
                                    f'Z: %{{z:.2f}}<br>' +
                                    '<extra></extra>'
                    ))
                    
            except Exception as e:
                # If OLS calculation fails, continue without trend line
                pass
        
        # Add color legend if color coding is used
        if color_column_idx is not None and color_column_idx != "" and color_text:
            # Create dummy traces for legend
            for i, unique_val in enumerate(unique_colors):
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color_palette[i % len(color_palette)]),
                    name=str(unique_val),
                    showlegend=True,
                    legendgroup=str(unique_val)
                ))
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title=csv_data['columns'][x_column_idx],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    title=csv_data['columns'][y_column_idx],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                zaxis=dict(
                    title=csv_data['columns'][z_column_idx],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                )
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Convert to JSON for frontend
        graph_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return JSONResponse(content={"plot": graph_json})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating 3D scatter plot: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)