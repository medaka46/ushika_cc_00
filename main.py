from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = FastAPI()

# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save-data")
async def save_data(request: Request):
    data = await request.json()
    return {"status": "success", "data": data}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
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
async def generate_scatter_plot(request: Request):
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
        print(f"DEBUG: OLS enabled: {ols_enabled}, x_numeric length: {len(x_numeric)}")
        if ols_enabled and len(x_numeric) >= 2:
            try:
                print(f"DEBUG: Starting OLS calculation with {len(x_numeric)} data points")
                
                # Prepare data for linear regression
                X = np.array(x_numeric).reshape(-1, 1)
                y = np.array(y_numeric)
                
                print(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}")
                print(f"DEBUG: X range: {np.min(X)} to {np.max(X)}")
                print(f"DEBUG: y range: {np.min(y)} to {np.max(y)}")
                
                # Check for invalid values
                if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)) or np.any(np.isinf(y)):
                    raise ValueError("Data contains NaN or infinite values")
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                print(f"DEBUG: Model fitted. Coef: {model.coef_[0]}, Intercept: {model.intercept_}")
                
                # Generate trend line points
                x_trend = np.linspace(min(x_numeric), max(x_numeric), 100)
                y_trend = model.predict(x_trend.reshape(-1, 1))
                
                # Calculate R-squared
                r_squared = model.score(X, y)
                
                print(f"DEBUG: R-squared: {r_squared}")
                
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
                print("DEBUG: OLS trend line added successfully")
            except Exception as e:
                # If OLS calculation fails, continue without trend line
                print(f"ERROR: Could not calculate OLS trend line: {str(e)}")
                print(f"ERROR: Exception type: {type(e).__name__}")
                import traceback
                print(f"ERROR: Full traceback: {traceback.format_exc()}")
        
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
async def generate_3d_scatter_plot(request: Request):
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
        
        # Add OLS regression plane if enabled for 3D plots
        print(f"DEBUG 3D: OLS enabled: {ols_enabled}, x_numeric length: {len(x_numeric)}")
        if ols_enabled and len(x_numeric) >= 3:
            try:
                print(f"DEBUG 3D: Starting 3D OLS calculation with {len(x_numeric)} data points")
                
                # Prepare data for multiple linear regression (Z = a*X + b*Y + c)
                X = np.column_stack([x_numeric, y_numeric])
                z = np.array(z_numeric)
                
                print(f"DEBUG 3D: X shape: {X.shape}, z shape: {z.shape}")
                print(f"DEBUG 3D: X range: [{np.min(X[:,0])}, {np.max(X[:,0])}], [{np.min(X[:,1])}, {np.max(X[:,1])}]")
                print(f"DEBUG 3D: z range: {np.min(z)} to {np.max(z)}")
                
                # Check for invalid values
                if np.any(np.isnan(X)) or np.any(np.isnan(z)) or np.any(np.isinf(X)) or np.any(np.isinf(z)):
                    raise ValueError("3D Data contains NaN or infinite values")
                
                # Fit multiple linear regression model
                model = LinearRegression()
                model.fit(X, z)
                
                print(f"DEBUG 3D: Model fitted. Coef: {model.coef_}, Intercept: {model.intercept_}")
                
                # Calculate R-squared
                r_squared = model.score(X, z)
                
                print(f"DEBUG 3D: R-squared: {r_squared}")
                
                # Create a mesh for the regression plane
                x_range = np.linspace(min(x_numeric), max(x_numeric), 20)
                y_range = np.linspace(min(y_numeric), max(y_numeric), 20)
                xx, yy = np.meshgrid(x_range, y_range)
                
                # Predict Z values for the mesh
                mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
                zz_pred = model.predict(mesh_points).reshape(xx.shape)
                
                # Check for invalid surface values
                if np.any(np.isnan(zz_pred)) or np.any(np.isinf(zz_pred)):
                    raise ValueError("3D Regression surface contains invalid values")
                
                # Add regression plane as a surface
                fig.add_trace(go.Surface(
                    x=xx.tolist(),
                    y=yy.tolist(),
                    z=zz_pred.tolist(),
                    opacity=0.3,
                    colorscale='Reds',
                    name=f'OLS Regression Plane (R² = {r_squared:.3f})',
                    showscale=False,
                    hovertemplate=f'<b>OLS Regression Plane</b><br>' +
                                f'R² = {r_squared:.3f}<br>' +
                                f'Equation: Z = {model.coef_[0]:.3f}*X + {model.coef_[1]:.3f}*Y + {model.intercept_:.3f}<br>' +
                                '<extra></extra>'
                ))
                print("DEBUG 3D: OLS regression plane added successfully")
            except Exception as e:
                # If OLS calculation fails, continue without regression plane
                print(f"ERROR 3D: Could not calculate 3D OLS regression plane: {str(e)}")
                print(f"ERROR 3D: Exception type: {type(e).__name__}")
                import traceback
                print(f"ERROR 3D: Full traceback: {traceback.format_exc()}")
        
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