import io
import pickle
from base64 import b64encode
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from matplotlib import pyplot as plt
from plotly.figure_factory import create_dendrogram
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn import linear_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, Dash, State, MATCH, ALL
import requests
from io import StringIO
import datetime
import gunicorn

# تابع بارگیری داده از گیت‌هاب
def load_data_from_github():
    url = "https://raw.githubusercontent.com/mostafatamandi/covid_dashboard/main/clean_df.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        print("داده با موفقیت از گیت‌هاب بارگیری شد")
        return data
    except Exception as e:
        print(f"خطا در بارگیری داده از گیت‌هاب: {e}")
        return None

# بارگیری اولیه داده
data = load_data_from_github()

# تعریف متغیرهای جهانی
descriptive_stats = pd.DataFrame()
results_df = pd.DataFrame()
best_model = None
best_model_name = ""
models = {}
roc_curves = {}
label_encoders = {}
target_encoder = LabelEncoder()
scaler = StandardScaler()
selected_features = []
heatmap_vars = []

imputer = SimpleImputer(strategy='mean')
if data is not None:
    missing_values = data[selected_features].isna().sum()

    for column in selected_features:
        if column in label_encoders:  # ویژگی کیفی
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
        else:  # ویژگی عددی
            mean_value = data[column].mean()
            data[column] = data[column].fillna(mean_value)

    if data[selected_features].isna().any().any():
        print("هشدار: همچنان مقادیر گمشده در داده‌ها وجود دارند!")

    # حذف ستون‌های مورد نظر
    columns_to_drop = ['PCR', 'Systole']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # تنظیم ستون‌های انتخابی
    selected_features = ["Age", "Hrct", "Naghs-imeni", "Palse_min",
                        "Temprature", "Gender", "Hb1", "NUT1", "O2sat"]
    target_column = "TEST"

    # بررسی وجود ستون‌های مورد نظر در داده‌ها
    missing_vars = [var for var in selected_features if var not in data.columns]
    if missing_vars:
        print(f"ستون‌های زیر در داده‌ها وجود ندارند: {', '.join(missing_vars)}")
        selected_features = [var for var in selected_features if var in data.columns]

    X = data[selected_features]
    y = data[target_column]

    # کدگذاری ویژگی‌های دسته‌ای
    for column in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

    y = target_encoder.fit_transform(y.astype(str))

    # استانداردسازی داده‌ها
    X = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X)

    # تقسیم داده‌ها به آموزش و آزمون
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # آموزش مدل‌ها
    models = {
        "Logistic Regression": linear_model.LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
            method='sigmoid', cv=3
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        )
    }

    # محاسبه معیارهای ارزیابی
    results = []
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(y_test)

        accuracy = round(accuracy_score(y_test, y_pred), 2)
        precision = round(precision_score(y_test, y_pred), 2)
        recall = round(recall_score(y_test, y_pred), 2)
        f1 = round(f1_score(y_test, y_pred), 2)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = round(auc(fpr, tpr), 2)
        roc_curves[name] = (fpr, tpr, roc_auc)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": roc_auc
        })

    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]

    # لیست متغیرهای مهم برای Heatmap
    heatmap_vars = [var for var in ["NUT1", "Age", "O2sat", "Temprature", "Hb1", "Palse_min"] if var in data.columns]

    # محاسبه آمار توصیفی
    descriptive_stats = data[selected_features].describe().loc[
        ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)

# توابع کمکی برای نمودارها
def create_roc_figure(roc_curves):
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

    for name, (fpr, tpr, auc_score) in roc_curves.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc_score})',
            mode='lines'
        ))

    fig.update_layout(
        title='نمودار منحنی مشخصه عملکرد برای مدل‌های مختلف',
        xaxis_title='نرخ مثبت کاذب',
        yaxis_title='نرخ مثبت واقعی',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500,
        template='plotly_white',
        font=dict(family='Vazir, Arial', size=14)
    )
    return fig

def create_gender_test_bar_chart():
    if data is None or 'Gender' not in data.columns or 'TEST' not in data.columns:
        return go.Figure()

    from plotly.subplots import make_subplots

    male_positive = data[(data['Gender'] == 'male') & (data['TEST'] == 'YES')].shape[0]
    male_negative = data[(data['Gender'] == 'male') & (data['TEST'] == 'NO')].shape[0]
    female_positive = data[(data['Gender'] == 'female') & (data['TEST'] == 'YES')].shape[0]
    female_negative = data[(data['Gender'] == 'female') & (data['TEST'] == 'NO')].shape[0]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]])

    # نمودار میله‌ای
    fig.add_trace(go.Bar(
        x=['مردان', 'زنان'],
        y=[male_positive, female_positive],
        name='مثبت',
        marker_color='#FF6B6B',
        text=[male_positive, female_positive],
        textposition='auto'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=['مردان', 'زنان'],
        y=[male_negative, female_negative],
        name='منفی',
        marker_color='#4ECDC4',
        text=[male_negative, female_negative],
        textposition='auto'
    ), row=1, col=1)

    # نمودار پای
    fig.add_trace(go.Pie(
        labels=['مردان مثبت', 'مردان منفی', 'زنان مثبت', 'زنان منفی'],
        values=[male_positive, male_negative, female_positive, female_negative],
        marker_colors=['#FF6B6B', '#4ECDC4', '#FF8787', '#7AE7C7'],
        hole=0.4
    ), row=1, col=2)

    fig.update_layout(
        title_text='',
        barmode='group',
        uniformtext_minsize=12,
        height=400,
        template='plotly_white',
        font=dict(family='Vazir, Arial', size=14)
    )
    return fig

def create_dendrogram(data, features, height=500, width=700):
    if data is None:
        return html.Div("داده‌ای برای نمایش وجود ندارد")

    try:
        X = data[features].copy()
        for col in X.select_dtypes(include=['object']):
            X[col] = LabelEncoder().fit_transform(X[col])

        dist_matrix = hierarchy.distance.pdist(X)
        linkage_matrix = hierarchy.linkage(dist_matrix, method='ward')

        fig = ff.create_dendrogram(
            X,
            orientation='bottom',
            linkagefun=lambda x: linkage_matrix,
            labels=data.index.astype(str).values,
            color_threshold=0.5 * max(linkage_matrix[:, 2])
        )
        fig.update_layout(
            title='',
            xaxis_title='شماره بیمار',
            yaxis_title='',
            width=width,
            height=height,
            hovermode='closest',
            template='plotly_white',
            font=dict(family='Vazir, Arial', size=12)
        )
        fig.update_xaxes(tickangle=90, tickfont=dict(size=10))
        return fig
    except Exception as e:
        print(f"خطا در ایجاد دندروگرام: {str(e)}")
        return go.Figure()

def get_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    else:
        return np.zeros(len(selected_features))

def get_top_features(model, feature_names, n=6):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return feature_names[:n]

    indices = np.argsort(importances)[-n:]
    return [feature_names[i] for i in indices]

def create_dynamic_input_form(top_features):
    form_fields = []
    for feat in top_features:
        if feat not in selected_features:
            continue

        if feat == 'Gender':
            options = [
                {"label": "زن", "value": "female"},
                {"label": "مرد", "value": "male"}
            ]
            field = dbc.Select(
                id={"type": "input-feature", "index": feat},
                options=options,
                value="female",
                required=True,
                className="form-control"
            )
        elif feat in ['Hrct', 'Naghs-imeni']:
            options = [
                {"label": "ندارد", "value": "NO"},
                {"label": "دارد", "value": "YES"}
            ]
            field = dbc.Select(
                id={"type": "input-feature", "index": feat},
                options=options,
                value="NO",
                required=True,
                className="form-control"
            )
        else:
            field = dbc.Input(
                id={"type": "input-feature", "index": feat},
                type="number",
                placeholder=f"مقدار {feat} را وارد کنید",
                step=0.01,
                required=True,
                min=0,
                className="form-control"
            )

        form_fields.append(
            dbc.Row([
                dbc.Label(feat, width=4, className="font-weight-bold"),
                dbc.Col(field, width=8)
            ], className="mb-3")
        )
    return form_fields

# ایجاد برنامه Dash با تم FLATLY
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# لایه‌های داشبورد
app.layout = html.Div([
    # هدر داشبورد
    html.Div([
        html.H1(
            "داشبورد تشخیص احتمال ابتلا به بیماری کووید-19",
            style={
                'text-align': 'center',
                'color': '#ffffff',
                'padding': '20px',
                'background': 'linear-gradient(90deg, #1A3C34 0%, #2A6F97 100%)',
                'border-radius': '10px 10px 0 0',
                'margin': '0'
            }
        ),
        html.Div([
            html.Button(
                "به‌روزرسانی داده‌ها",
                id="refresh-button",
                n_clicks=0,
                style={
                    'margin': '10px',
                    'padding': '10px 20px',
                    'font-size': '16px',
                    'background-color': '#28A745',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '5px',
                    'cursor': 'pointer',
                    'transition': 'background-color 0.3s'
                },
                className="hover:bg-green-700"
            ),
            html.Div(
                id="last-update",
                style={
                    'margin': '10px',
                    'font-size': '14px',
                    'color': '#ffffff',
                    'background': '#2A6F97',
                    'padding': '10px',
                    'border-radius': '5px'
                }
            )
        ], style={'text-align': 'center', 'background': '#2A6F97', 'padding-bottom': '10px'})
    ]),

    # تب‌ها
    dcc.Tabs(id="tabs", value="tab-descriptive", children=[
        # تب آمار توصیفی
        dcc.Tab(label="آمار توصیفی", value="tab-descriptive", children=[
            html.Div([
                dbc.Card([
                    dbc.CardHeader("آمار توصیفی داده‌ها", className="text-center h4"),
                    dbc.CardBody([
                        dbc.Table.from_dataframe(
                            descriptive_stats.reset_index(),
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True,
                            style={
                                'font-size': '14px',
                                'text-align': 'center'
                            }
                        )
                    ])
                ], className="shadow-sm mb-4"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("نمودار همبستگی",  className="text-center h4"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='correlation-heatmap',
                                    figure=px.imshow(
                                        data[heatmap_vars].corr() if data is not None else np.zeros((4, 4)),
                                        labels=dict(x="متغیرها", y="متغیرها", color="همبستگی"),
                                        x=heatmap_vars,
                                        y=heatmap_vars,
                                        color_continuous_scale='Viridis',
                                        zmin=-1,
                                        zmax=1
                                    ).update_layout(
                                        title="",
                                        width=500,
                                        height=500,
                                        template='plotly_white',
                                        font=dict(family='Vazir, Arial', size=14)
                                    ) if data is not None else {}
                                )
                            ])
                        ], className="shadow-sm")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("تعداد موارد مثبت و منفی بر اساس جنسیت", className="text-center h4"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='gender-test-chart',
                                    figure=create_gender_test_bar_chart()
                                )
                            ])
                        ], className="shadow-sm")
                    ], md=6)
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("نمودار خوشه‌بندی بیماران", className="text-center h4"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='dendrogram-graph',
                            config={'displayModeBar': True}
                        )
                    ])
                ], className="shadow-sm mb-4"),

                html.H3("هیستوگرام و نمودار جعبه‌ای متغیرهای مهم", className="text-center mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"هیستوگرام و جعبه‌ای {var}", className="text-center"),
                            dbc.CardBody([
                                dcc.Graph(
                                    figure={
                                        'data': [go.Histogram(x=data[var], name=var)],
                                        'layout': go.Layout(
                                            title='',
                                            xaxis_title="مقدار",
                                            yaxis_title="تعداد",
                                            template='plotly_white'
                                        )
                                    }
                                ),
                                dcc.Graph(
                                    figure={
                                        'data': [
                                            go.Box(
                                                x=data['Gender'] if 'Gender' in data.columns else ['All'] * len(data),
                                                y=data[var],
                                                name=var,
                                                boxpoints='all',
                                                jitter=0.3,
                                                pointpos=-1.8
                                            )
                                        ],
                                        'layout': go.Layout(
                                            title='',
                                            yaxis_title="مقدار",
                                            xaxis_title="جنسیت" if 'Gender' in data.columns else '',
                                            template='plotly_white'
                                        )
                                    }
                                )
                            ])
                        ], className="shadow-sm mb-4")
                    ], md=6) for var in heatmap_vars if data is not None and var in data.columns
                ])
            ], style={'padding': '20px', 'background': '#F8F9FA'})
        ]),

        # تب برازش مدل
        dcc.Tab(label="برازش مدل", value="tab-modeling", children=[
            html.Div([
                dbc.Card([
                    dbc.CardHeader("جدول مقایسه مدل‌ها", className="text-center h4"),
                    dbc.CardBody([
                        dbc.Table.from_dataframe(
                            results_df,
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True,
                            style={
                                'font-size': '14px',
                                'text-align': 'center'
                            }
                        ) if not results_df.empty else html.Div("داده‌ای برای نمایش وجود ندارد", className="text-center")
                    ])
                ], className="shadow-sm mb-4"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("بهترین مدل بر اساس دقت", className="text-center h4"),
                            dbc.CardBody([
                                html.Div(
                                    f"بهترین مدل: {best_model_name} (دقت: {results_df.loc[results_df['Accuracy'].idxmax(), 'Accuracy']})" if not results_df.empty else "هیچ مدلی آموزش داده نشده",
                                    style={
                                        'font-size': '18px',
                                        'text-align': 'center',
                                        'padding': '10px',
                                        'background': '#E9ECEF',
                                        'border-radius': '5px'
                                    }
                                )
                            ])
                        ], className="shadow-sm")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("معیارهای ارزیابی بهترین مدل", className="text-center h4"),
                            dbc.CardBody([
                                dbc.Table(
                                    [
                                        html.Tr([html.Td("دقت (Accuracy)"), html.Td(results_df.loc[results_df['Accuracy'].idxmax(), 'Accuracy'])]),
                                        html.Tr([html.Td("Precision"), html.Td(results_df.loc[results_df['Accuracy'].idxmax(), 'Precision'])]),
                                        html.Tr([html.Td("Recall (حساسیت)"), html.Td(results_df.loc[results_df['Accuracy'].idxmax(), 'Recall'])]),
                                        html.Tr([html.Td("F1-Score"), html.Td(results_df.loc[results_df['Accuracy'].idxmax(), 'F1-Score'])]),
                                        html.Tr([html.Td("AUC"), html.Td(results_df.loc[results_df['Accuracy'].idxmax(), 'AUC'])])
                                    ],
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    style={'font-size': '14px', 'text-align': 'center'}
                                ) if not results_df.empty else html.Div("داده‌ای برای نمایش وجود ندارد", className="text-center")
                            ])
                        ], className="shadow-sm")
                    ], md=6)
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("مقایسه اهمیت متغیرها در مدل‌های مختلف", className="text-center h4"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='model-selector',
                            options=[{'label': name, 'value': name} for name in models.keys()],
                            value=list(models.keys())[0] if models else None,
                            multi=True,
                            placeholder='مدل‌ها را انتخاب کنید',
                            className="mb-3"
                        ),
                        dcc.Graph(id='feature-importance-comparison')
                    ])
                ], className="shadow-sm mb-4"),

                dbc.Card([
                    dbc.CardHeader("نمودار مشخصه عملکرد مدل‌ها", className="text-center h4"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='roc-curve',
                            figure=create_roc_figure(roc_curves) if roc_curves else {}
                        )
                    ])
                ], className="shadow-sm mb-4")
            ], style={'padding': '20px', 'background': '#F8F9FA'})
        ]),

        # تب پیش‌بینی
        dcc.Tab(label="🔮 پیش‌بینی", value="tab-prediction", children=[
            html.Div([
                dcc.Store(id="model-store"),
                dcc.Store(id="top-features-store"),
                dbc.Card([
                    dbc.CardHeader("پیش‌بینی وضعیت بیمار", className="text-center h4"),
                    dbc.CardBody([
                        html.Div(id="prediction-form-container"),
                        dbc.Button(
                            "انجام پیش‌بینی",
                            id="predict-btn",
                            color="primary",
                            className="mt-3 w-100",
                            style={
                                'font-size': '16px',
                                'padding': '10px',
                                'border-radius': '5px',
                                'transition': 'background-color 0.3s'
                            }
                        ),
                        html.Div(id="prediction-result", className="mt-4")
                    ])
                ], className="shadow-sm")
            ], style={'padding': '20px', 'background': '#F8F9FA', 'max-width': '600px', 'margin': 'auto'})
        ])
    ], style={'background': '#FFFFFF', 'border-radius': '10px', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'})
], style={
    'min-height': '100vh',
    'background': 'linear-gradient(135deg, #E9ECEF 0%, #F8F9FA 100%)',
    'padding': '0'
})

# Callback برای به‌روزرسانی داده‌ها
@app.callback(
    Output("last-update", "children"),
    [Input("refresh-button", "n_clicks")]
)
def update_data(n_clicks):
    if n_clicks > 0:
        global data, X, y, X_train, X_test, y_train, y_test, models, results_df, best_model, best_model_name, descriptive_stats, roc_curves

        new_data = load_data_from_github()
        if new_data is not None:
            data = new_data
            data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
            X = data[selected_features]
            y = data[target_column]

            for column in X.select_dtypes(include=["object"]).columns:
                le = label_encoders.get(column, LabelEncoder())
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

            y = target_encoder.fit_transform(y)
            X = imputer.fit_transform(X)
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = []
            roc_curves = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(y_test)

                accuracy = round(accuracy_score(y_test, y_pred), 2)
                precision = round(precision_score(y_test, y_pred), 2)
                recall = round(recall_score(y_test, y_pred), 2)
                f1 = round(f1_score(y_test, y_pred), 2)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = round(auc(fpr, tpr), 2)
                roc_curves[name] = (fpr, tpr, roc_auc)

                results.append({
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "AUC": roc_auc
                })

            results_df = pd.DataFrame(results)
            best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
            best_model = models[best_model_name]

            descriptive_stats = data[selected_features].describe().loc[
                ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)

            return f"آخرین به‌روزرسانی: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - داده‌ها با موفقیت به‌روزرسانی شدند"
        else:
            return f"آخرین به‌روزرسانی: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - خطا در بارگیری داده‌ها"

    return f"آخرین به‌روزرسانی: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - برای به‌روزرسانی دکمه را بزنید"

# Callback برای مقایسه اهمیت متغیرها
@app.callback(
    Output('feature-importance-comparison', 'figure'),
    [Input('model-selector', 'value')]
)
def update_feature_importance_comparison(selected_models):
    if not selected_models or not models:
        return go.Figure()

    if not isinstance(selected_models, list):
        selected_models = [selected_models]

    data = []
    for model_name in selected_models:
        if model_name in models:
            model = models[model_name]
            importance = get_feature_importance(model, model_name)
            data.append(
                go.Bar(
                    x=selected_features,
                    y=importance,
                    name=model_name,
                    marker_color='#FF6B6B' if model_name == best_model_name else '#4ECDC4'
                )
            )

    return {
        'data': data,
        'layout': go.Layout(
            title='',
            xaxis_title='متغیرها',
            yaxis_title='میزان اهمیت',
            barmode='group',
            height=400,
            template='plotly_white',
            font=dict(family='Vazir, Arial', size=14)
        )
    }

@app.callback(
    Output('dendrogram-graph', 'figure'),
    Input('dendrogram-graph', 'id')
)
def update_dendrogram(_):
    return create_dendrogram(
        data=data,
        features=selected_features
    )

# Callback برای پیش‌بینی
@app.callback(
    [Output("prediction-form-container", "children"),
     Output("top-features-store", "data")],
    Input("tabs", "value")
)
def update_prediction_form(tab_value):
    if tab_value != "tab-prediction":
        raise PreventUpdate

    if not best_model or not selected_features:
        return html.Div("مدلی برای پیش‌بینی وجود ندارد یا ویژگی‌ها تعریف نشده‌اند", className="text-center"), []

    try:
        top_features = get_top_features(best_model, selected_features, n=6)
        if not top_features:
            return html.Div("هیچ ویژگی مهمی یافت نشد", className="text-center"), []

        form = create_dynamic_input_form(top_features)
        return form, top_features
    except Exception as e:
        print(f"خطا در تولید فرم پیش‌بینی: {str(e)}")
        return html.Div(f"خطا در تولید فرم: {str(e)}", className="text-center"), []

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "input-feature", "index": ALL}, "value"),
    State({"type": "input-feature", "index": ALL}, "id"),
    State("top-features-store", "data"),
    prevent_initial_call=True
)
def make_prediction(n_clicks, values, input_ids, top_features):
    if not n_clicks or not top_features or not values or len(values) != len(top_features):
        raise PreventUpdate

    try:
        if any(v is None or (isinstance(v, str) and v.strip() == "") for v in values):
            print("مقادیر ورودی:", values)
            return dbc.Alert("لطفاً تمام فیلدها را پر کنید.", color="warning", className="text-center")

        feature_names = [input_id['index'] for input_id in input_ids]
        input_data = pd.DataFrame([values], columns=feature_names)

        for column in input_data.columns:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    input_data[column] = le.transform(input_data[column])
                except ValueError as e:
                    print(f"خطا در کدگذاری {column}: {str(e)}")
                    return dbc.Alert(f"مقدار نامعتبر برای {column}: {input_data[column].values[0]}", color="warning", className="text-center")

        full_input = pd.DataFrame(columns=selected_features, index=[0], dtype=float)
        for feature in feature_names:
            if feature in selected_features:
                full_input[feature] = input_data[feature].astype(float)

        for feature in selected_features:
            if feature not in feature_names:
                if feature in label_encoders:
                    mode_value = data[feature].mode()[0]
                    if pd.isna(mode_value):
                        return dbc.Alert(f"داده‌های اصلی برای {feature} شامل مقادیر گمشده هستند", color="warning", className="text-center")
                    full_input[feature] = float(label_encoders[feature].transform([mode_value])[0])
                else:
                    mean_value = data[feature].mean()
                    if pd.isna(mean_value):
                        return dbc.Alert(f"داده‌های اصلی برای {feature} شامل مقادیر گمشده هستند", color="warning", className="text-center")
                    full_input[feature] = float(mean_value)

        if full_input.isna().any().any():
            print("NaN در full_input شناسایی شد:")
            print(full_input)
            return dbc.Alert("داده‌های ورودی شامل مقادیر گمشده هستند", color="warning", className="text-center")

        input_array = imputer.transform(full_input[selected_features].values)
        input_array = scaler.transform(input_array)

        prediction = best_model.predict(input_array)
        proba = best_model.predict_proba(input_array)[0]

        result = "مثبت" if prediction[0] == 1 else "منفی"
        probability = proba[1] * 100 if prediction[0] == 1 else proba[0] * 100

        return dbc.Card([
            dbc.CardBody([
                html.H4(
                    f"نتیجه: {result} (احتمال: {probability:.1f}%)",
                    className="text-center",
                    style={'color': '#FF6B6B' if prediction[0] == 1 else '#4ECDC4'}
                ),
                html.P(
                    "لطفاً برای تشخیص دقیق‌تر با پزشک مشورت کنید.",
                    className="text-center mt-2"
                )
            ])
        ], className="shadow-sm")
    except Exception as e:
        print(f"خطا در پیش‌بینی: {str(e)}")
        return dbc.Alert(f"خطا در پیش‌بینی: {str(e)}", color="warning", className="text-center")
#if __name__ == '__main__':
#    app.run_server(host="0.0.0.0", port=8000, debug=False)       
        
