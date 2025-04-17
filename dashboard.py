# Enhanced Dash App with Visualization Toggle and ML Algorithm Comparison
import dash
from dash import html, dcc, Input, Output, State, dash_table,ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import seaborn as sns


# Load actual student performance dataset
df = pd.read_csv("Student_performance_data .csv")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example results from different ML models
ml_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLPClassifier'],
    'Train Accuracy':[0.829, 0.999, 0.994, 0.914],
    'Test Accuracy': [0.816, 0.918, 0.912, 0.910]
})

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# -------------------- Layout -------------------- #
app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id="theme-container", className="main", children=[
        html.Div([
            html.H2("\ud83d\udcca Dashboard", className="title"),
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('Analytics', href='/analytics', className='nav-link'),
            dcc.Link('Info Page', href='/info', className='nav-link'),
            dcc.Link('Prediction', href='/predict', className='nav-link'),
            dcc.Link('Feedback', href='/feedback', className='nav-link'),
            html.Button("\ud83c\udf19 Toggle Dark Mode", id="dark-toggle", n_clicks=0, className='toggle'),
        ], className="sidebar"),
        html.Div(id='page-content', className="main-content")
    ])
])

# -------------------- Page Content Callback -------------------- #
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/analytics':
        return analytics_layout
    elif pathname == '/info':
        return info_layout
    elif pathname == '/predict':
        return prediction_layout
    elif pathname == '/feedback':
        return feedback_layout
    else:
        return home_layout

# -------------------- Page Layouts -------------------- #
home_layout = html.Div([
    html.H1("Welcome to our Machine Learning 382 Project ! ", className="page-title"),
    html.H3("Group members are:", className="page-desc"),
    html.Ul([
        html.Li("Trent Evans"),
        html.Li("Demica Smit"),
        html.Li("Bianca Long"),
        html.Li("Jade Riley")
    ]),
    html.P("In this project we used machine learning & deep algorithms to solve the BrightPath Academy problem..."),
    html.Button("📄 Generate summary Report", id="generate-report-btn", n_clicks=0, className="btn"),
    dcc.Download(id="download-report"),
    html.Button("📄 Generate Bright path accademy Report", id="generate-report-btn2", n_clicks=0, className="btn"),
    dcc.Download(id="download-report2"),

])

analytics_layout = html.Div([
    html.H2("Data Visualization Selector", className="section-title"),
    dcc.Dropdown(
        id='viz-type',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Box Plot', 'value': 'box'},
            {'label': 'Heatmap', 'value': 'heatmap'}
        ],
        value='scatter',
        className='input',
        style={'color': 'black'}
    ),
    dcc.Graph(id='dynamic-graph'),

    html.H2("Comparision of our ML Algorithms", className="section-title"),
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(name='Train Accuracy', x=ml_results['Model'], y=ml_results['Train Accuracy']),
                go.Bar(name='Test Accuracy', x=ml_results['Model'], y=ml_results['Test Accuracy'])
            ],
            layout=go.Layout(
                barmode='group',
                template='plotly_dark',
                title='Comparison of ML Model Performance'
            )
        )
    )
])

info_layout = html.Div([
    html.H2("Dataset Information & Summary", className="section-title"),
    html.Div([
        html.H4("Preview of Dataset:"),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_data={"color": "#000000"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4", "color": "#000000"},
            style_data_conditional=[
                {
                    "if": {"state": "selected"},
                    "backgroundColor": "#d0d0d0",
                    "color": "#000000"
                }
            ],
            page_size=10
        ),

        html.H4("Statistical Summary of cleaned dataset:"),
        html.Div(id="summary-output")
    ])
])

feedback_layout = html.Div([
    html.H2("User Feedback Form", className="section-title"),
    html.Div([
        dcc.Input(id='username', type='text', placeholder='Your Name', className='input'),
        dcc.Textarea(id='comments', placeholder='Enter feedback...', className='input', style={'height': '100px'}),
        html.Button('Submit', id='submit-feedback', n_clicks=0, className='btn'),
        html.Div(id='feedback-result')
    ])
])

prediction_layout = html.Div([
    html.H2("Student Performance Prediction", className="section-title"),
    html.Div([
        dcc.Input(id='input-studytime', type='number', placeholder='Study Time Weekly (hrs)', className='input'),
        dcc.Input(id='input-gpa', type='number', placeholder='GPA', className='input'),
        html.Button('Predict Grade Class', id='predict-btn', n_clicks=0, className='btn'),
        html.Div(id='prediction-output', style={'marginTop': '20px'})
    ])
])


# -------------------- CSS -------------------- #
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>MLG 382 - Project 1</title>
    {%favicon%}
    {%css%}
    <style>
        body, html { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        .main { display: flex; flex-direction: row; min-height: 100vh; background-color: white; color: black; }
        .main.dark { background-color: #121212; color: white; }
        .sidebar { width: 220px; background-color: #2c3e50; color: white; padding: 20px; }
        .main.dark .sidebar { background-color: #1e1e1e; color: white; }
        .nav-link { display: block; color: white; margin: 10px 0; text-decoration: none; font-weight: bold; }
        .nav-link:hover { color: #18bc9c; }
        .toggle { margin-top: 30px; padding: 10px; background: #34495e; border: none; color: white; cursor: pointer; width: 100%; }
        .main-content { flex: 1; padding: 40px; background-color: inherit; color: inherit; }
        .page-title { font-size: 2rem; }
        .section-title { font-size: 1.5rem; margin-bottom: 20px; }
        .input { display: block; width: 100%; max-width: 500px; margin-bottom: 10px; padding: 8px; font-size: 1rem; color: black; }
        .main.dark .input { color: white; background-color: #333; border: 1px solid #555; }
        .main.dark .dash-dropdown .Select-control, .main.dark .dash-dropdown .Select-menu-outer {
            background-color: #333;
            color: white;
        }
        .main.dark .dash-dropdown .Select-placeholder,
        .main.dark .dash-dropdown .Select-value-label {
            color: white !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# -------------------- Callbacks -------------------- #
@app.callback(
    Output("theme-container", "className"),
    Input("dark-toggle", "n_clicks")
)
def toggle_dark_mode(n_clicks):
    return "main dark" if n_clicks % 2 else "main"

@app.callback(
    Output('feedback-result', 'children'),
    Input('submit-feedback', 'n_clicks'),
    State('username', 'value'),
    State('comments', 'value'),
    prevent_initial_call=True
)
def submit_feedback(n_clicks, name, comment):
    if not name or not comment:
        return html.Div("\u274c Please complete all fields.", style={'color': 'red'})
    return html.Div(f"\u2705 Thank you, {name}, for your feedback!", style={'color': 'green'})

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-studytime', 'value'),
    State('input-gpa', 'value'),
    prevent_initial_call=True
)
def predict_grade(n_clicks, studytime, gpa, ):
    if studytime is None or gpa is None:
        return html.Div("Please fill in all fields.", style={'color': 'red'})

    try:
        # Create DataFrame for prediction
        new_data = pd.DataFrame({
            "StudyTimeWeekly": [studytime],
            "Absences": [0],  # Placeholder, if Absences is needed in scaling
            "GPA": [gpa]
        })

        # Scale input
        scaled_input = scaler.transform(new_data)
        scaled_df = pd.DataFrame(scaled_input, columns=["StudyTimeWeekly", "Absences", "GPA"])
        input_for_model = scaled_df[["StudyTimeWeekly", "GPA"]]

        # Make prediction
        prediction = rf_model.predict(input_for_model)[0]

        return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'green'})
        ])
    except Exception as e:
        return html.Div(f"Error making prediction: {e}", style={'color': 'red'})

@app.callback(
    Output('dynamic-graph', 'figure'),
    Input('viz-type', 'value')
)
def update_graph(viz_type):
    if viz_type == 'scatter':
        return px.scatter(df, x="StudyTimeWeekly", y="GPA", color=df['GradeClass'].astype(str), template="plotly_dark")
    elif viz_type == 'bar':
        return px.bar(df, x="ParentalEducation", y="GPA", color=df['GradeClass'].astype(str), template="plotly_dark")
    elif viz_type == 'box':
        return px.box(df, x="Gender", y="GPA", color=df['GradeClass'].astype(str), template="plotly_dark")
    elif viz_type == 'heatmap':
        return px.imshow(df[['GradeClass', 'StudyTimeWeekly', 'Absences', 'GPA']].corr(), text_auto=True, color_continuous_scale='Blues', template="plotly_dark")
    else:
        return go.Figure()


@app.callback(
    Output('summary-output', 'children'),
    Input('theme-container', 'className')
)
def update_summary(theme_class):
    color = 'white' if 'dark' in theme_class else 'black'
    return html.Pre(df.describe().T.to_string(), style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'color': color})

@app.callback(
    Output("download-report", "data"),
    Input("generate-report-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_summary_report(n_clicks):
    return dcc.send_file("summaryReport.pdf")

@app.callback(
    Output("download-report2", "data"),
    Input("generate-report-btn2", "n_clicks"),
    prevent_initial_call=True
)
def download_brightpath_report(n_clicks):
    return dcc.send_file("BrightPath_Report.pdf")

# -------------------- Run App -------------------- #
if __name__ == "__main__":
    app.run(debug=True)
