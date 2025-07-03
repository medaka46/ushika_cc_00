from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.utils

app = FastAPI()

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
        
        table_data = {
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
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
        csv_data = data.get('csv_data')
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No CSV data provided")
        
        # Extract data for plotting
        x_values = [row[x_column_idx] for row in csv_data['data'] if row[x_column_idx] is not None]
        y_values = [row[y_column_idx] for row in csv_data['data'] if row[y_column_idx] is not None]
        
        # Extract hover column data if specified
        hover_values = []
        if hover_column_idx is not None and hover_column_idx != "":
            hover_column_idx = int(hover_column_idx)
            hover_values = [row[hover_column_idx] for row in csv_data['data']]
        
        # Convert to numeric, filter out non-numeric values
        x_numeric = []
        y_numeric = []
        hover_text = []
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
        
        # Create Plotly scatter plot
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
        
        fig.update_layout(
            title=dict(
                text=f"Interactive Scatter Plot: {csv_data['columns'][y_column_idx]} vs {csv_data['columns'][x_column_idx]}",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)