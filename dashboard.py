
import dash
from dash import html, dcc, Input, Output, State, dash_table,ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import seaborn as sns
import datetime


# Load actual student performance dataset
df = pd.read_csv("DataSets/Student_performance_data .csv")
rf_model = joblib.load("BestModel/rf_model.pkl")
scaler = joblib.load("BestModel/scaler.pkl")

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
    html.Div(id="theme-container", className="main dark", children=[
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
    html.Div(className="hero-section", children=[
        html.H1("🚀 BrightPath Academy - ML Project", className="hero-title"),
        html.P("Predicting and enhancing student performance using machine learning algorithms.", className="hero-subtitle"),
    ]),
    html.Div(className="team-section", children=[
        html.H2("👨‍👩‍👧‍👦 Meet the Team", className="section-title"),
        html.Div(className="team-grid", children=[
            html.Div(className="card", children=[html.H4("Trent Evans"), html.P("600383")]),
            html.Div(className="card", children=[html.H4("Demica Smit"), html.P("577875")]),
            html.Div(className="card", children=[html.H4("Bianca Long"), html.P("600476")]),
            html.Div(className="card", children=[html.H4("Jade Riley"), html.P("578125")]),
        ])
    ]),
    html.Div(className="summary-section", children=[
        html.H2("📄 Project Summary", className="section-title"),
        html.P("We built a predictive ML model to help BrightPath Academy identify students at risk and provide timely intervention."),
        html.Div(className="button-group", children=[
            html.Button("📄 Download Summary Report", id="generate-report-btn", n_clicks=0, className="btn"),
            dcc.Download(id="download-report"),
            html.Button("📄 BrightPath Academy Report", id="generate-report-btn2", n_clicks=0, className="btn"),
            dcc.Download(id="download-report2"),
        ])
    ])
], className="center-content")

analytics_layout = html.Div([
    html.Div(className='card-section', children=[
        html.H2("Data Visualization Selector", className="section-title"),
        
html.Div(className='dash-dropdown', children=[
    dcc.Dropdown(
        id='viz-type',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Box Plot', 'value': 'box'},
            {'label': 'Heatmap', 'value': 'heatmap'}
        ],
        value='scatter',
        className='input'
    )
])
,
        dcc.Graph(id='dynamic-graph')
    ]),
    html.Div(className='card-section', children=[
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
])

info_layout = html.Div([
    html.Div(className='card-section', children=[
        html.H2("Dataset Information & Summary", className="section-title"),
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
    html.Div(className='card-section feedback-container', children=[
        html.H2("User Feedback Form", className="section-title"),
        dcc.Input(id='username', type='text', placeholder='Your Name', className='input'),
        dcc.Textarea(id='comments', placeholder='Enter feedback...', className='input', style={'height': '100px'}),
        html.Button('Submit', id='submit-feedback', n_clicks=0, className='btn'),
        html.Div(id='feedback-result', className='feedback-result')
    ])
])

prediction_layout = html.Div([
    html.Div(className='card-section', children=[
        html.H2("Student Performance Prediction - Using our best performing model (RandomForest)", className="section-title"),
        html.Div(className='input-grid', children=[
            dcc.Input(id='input-studytime', type='number', placeholder='Study Time Weekly (hrs)', className='input'),
            dcc.Input(id='input-gpa', type='number', placeholder='GPA', className='input'),
            dcc.Input(type='number', placeholder='Age', className='input'),
            dcc.Input(type='number', placeholder='Absenses', className='input'),
            dcc.Input(type='number', placeholder='Extracurricular', className='input')
        ]),
        html.Button('Predict Grade Class', id='predict-btn', n_clicks=0, className='btn'),
        html.Div(id='prediction-output', className='card-section', style={'marginTop': '20px'})
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
        body, html { margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .main { display: flex; flex-direction: row; min-height: 100vh; background-color: white; color: black; }
        .main.dark { background-color: #121212; color: white; }
        .sidebar {
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease; width: 220px; background-color: #2c3e50; color: white; padding: 20px; }
        .main.dark .sidebar {
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease; background-color: #1e1e1e; color: white; }
        .nav-link { display: block; color: white; margin: 10px 0; text-decoration: none; font-weight: bold; }
        .nav-link:hover { color: #18bc9c; }
        .toggle {
            border-radius: 6px;
            transition: background 0.3s ease;
            background-color: #1abc9c; margin-top: 30px; padding: 10px; background: #34495e; border: none; color: white; cursor: pointer; width: 100%; }
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
        .center-content {
            transition: all 0.3s ease;
            transform: translateY(0);
            animation: fadeIn 0.5s ease-in;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 40px;
            border-radius: 10px;
            max-width: 900px;
            margin: 20px auto 50px auto;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
            background-color: rgba(255, 255, 255, 0.85); /* default light */
    border-radius: 10px;
    max-width: 900px;
    margin: 0px auto 50px auto;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.1);

}


    
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    
        .hero-section {
            text-align: center;
            padding: 20px;
            margin-bottom: 40px;
        }
        .hero-title {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            color: #555;
        }
        .team-section {
            max-width: 800px;
            margin: 0 auto 40px auto;
        }
        .team-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .main.dark .card {
            background-color: #2e2e2e;
            color: white;
        }
        .summary-section {
            margin-top: 40px;
            text-align: center;
        }
    
        .card-section {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .main.dark .center-content {
            background-color: rgba(30, 30, 30, 0.95);
        }
        .main.dark .card-section {
            background-color: #1f1f1f;
        }
        .input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .dash-table-container {
            margin-top: 20px;
        }
        .feedback-container {
            max-width: 600px;
            margin: auto;
        }
        .feedback-result {
            margin-top: 15px;
            font-weight: bold;
        }

    
        .btn {
            padding: 12px 18px;
            margin: 5px;
            font-size: 1rem;
            background: linear-gradient(to right, #1abc9c, #16a085);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
        }
        .btn:hover {
            background: linear-gradient(to right, #16a085, #1abc9c);
            transform: scale(1.03);
        }

    
.main.dark .dash-dropdown .Select-control {
    background-color: #333 !important;
    color: white !important;
    border-color: #555;
}

.main.dark .dash-dropdown .Select-menu-outer {
    background-color: #333 !important;
    color: white !important;
}

.main.dark .dash-dropdown .Select-value-label,
.main.dark .dash-dropdown .Select-placeholder {
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
    return "main" if n_clicks % 2 else "main dark"

@app.callback(
    Output('feedback-result', 'children'),
    Input('submit-feedback', 'n_clicks'),
    State('username', 'value'),
    State('comments', 'value'),
    prevent_initial_call=True
)
def submit_feedback(n_clicks, name, comment):
    if not name or not comment:
        return html.Div("❌ Please complete all fields.", style={'color': 'red'})

    # Append feedback to a text file
    try:
        with open("user_feedback.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Feedback Submitted on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Name: {name}\n")
            f.write(f"Comment: {comment}\n\n")
    except Exception as e:
        return html.Div(f"⚠️ Error saving feedback: {e}", style={'color': 'red'})

    return html.Div(f"✅ Thank you, {name}, for your feedback!", style={'color': 'green'})

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


        if prediction == 0:
            score = "A - Student is doing great"
            return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'green'}),
            html.H4(f"Student official grade: {score}")
        ])
        elif prediction == 1:
            score = "B - Student is doing well"
            return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'green'}),
            html.H4(f"Student official grade: {score}")
        ])
        elif prediction == 2:
            score = "C - Student is Average"
            return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'green'}),
            html.H4(f"Student official grade: {score}")
        ])
        elif prediction == 3:
            score = "D - Student is on the verge of failing"
            return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'green'}),
            html.H4(f"Student official grade: {score}")
        ])
        elif prediction == 4:
            score = "F - Student has failed, Please seek student extra help !!!"
            return html.Div([
            html.H4(f"Predicted Grade Class: {prediction}", style={'color': 'red'}),
            html.H4(f"Student official grade: {score}")
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
    return dcc.send_file("Reports/summaryReport.pdf")

@app.callback(
    Output("download-report2", "data"),
    Input("generate-report-btn2", "n_clicks"),
    prevent_initial_call=True
)
def download_brightpath_report(n_clicks):
    return dcc.send_file("Reports/BrightPathAcademy_Report.pdf")

# -------------------- Run App -------------------- #
if __name__ == "__main__":
    app.run(debug=True)
