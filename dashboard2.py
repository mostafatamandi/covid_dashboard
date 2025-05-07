import streamlit as st
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn import linear_model
from sklearn.calibration import CalibratedClassifierCV
from scipy.cluster.hierarchy import linkage
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, no_update, callback
import dash_bootstrap_components as dbc
import requests
from io import StringIO
import datetime
from plotly import figure_factory as ff


# ============================================
# 1. توابع کمکی و تنظیمات اولیه
# ============================================

def load_data():
    """بارگیری داده از گیت‌هاب"""
    try:
        url = "https://raw.githubusercontent.com/mostafatamandi/covid_dashboard/main/clean_df.csv"
        data = pd.read_csv(StringIO(requests.get(url).text))
        print("داده با موفقیت بارگیری شد")
        return data.drop(columns=['PCR', 'Systole'], errors='ignore')
    except Exception as e:
        print(f"خطا در بارگیری داده: {e}")
        return None


def prepare_data(data):
    """آماده‌سازی داده‌ها"""
    features = ["Hrct","NUT1","Naghs-imeni", "Palse_min", "Temprature", "O2sat", "Gender", "Hb1","Age", "Cough","Breath_min"]
    target = "TEST"

    X = data[features]
    y = data[target]

    # کدگذاری متغیرهای کیفی
    encoders = {}
    for col in X.select_dtypes(include=['object']):
        encoders[col] = LabelEncoder().fit(X[col])
        X[col] = encoders[col].transform(X[col])

    y = LabelEncoder().fit_transform(y)
    X = StandardScaler().fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42), encoders, features


def train_models(X_train, y_train):
    """آموزش مدل‌های مختلف"""
    models = {
        "Logistic Regression": linear_model.LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": CalibratedClassifierCV(SVC(probability=True)),
        "XGBoost": XGBClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    """ارزیابی مدل‌ها"""
    results = []
    roc_data = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1": round(f1_score(y_test, y_pred), 3),
            "AUC": round(auc(*roc_curve(y_test, y_proba)[:2]), 3)
        })

        roc_data[name] = roc_curve(y_test, y_proba)

    return pd.DataFrame(results), roc_data


def get_feature_description(feature):
    """توضیحات متغیرها"""
    descriptions = {
        "Hrct": "نتیجه HRCT (0: طبیعی, 1: غیرطبیعی)",
        "Naghs-imeni": "وضعیت نقص ایمنی (0: ندارد, 1: دارد)",
        "Palse_min": "ضربان قلب در دقیقه (معمولاً بین 60 تا 100)",
        "Temprature": "دمای بدن به درجه سانتیگراد (معمولاً بین 35 تا 42)",
        "NUT": "نتیجه تست NUT",
        "O2sat": "اشباع اکسیژن خون (%) (معمولاً بین 90 تا 100)"
    }
    return descriptions.get(feature, "لطفاً مقدار معتبر وارد کنید")


def get_feature_importance(model, model_name):
    """استخراج اهمیت متغیرها از مدل"""
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    else:
        return np.zeros(len(selected_features))


# ============================================
# 2. آماده‌سازی داده و مدل‌ها
# ============================================

raw_data = load_data()
if raw_data is not None:
    (X_train, X_test, y_train, y_test), label_encoders, feature_names = prepare_data(raw_data)
    models = train_models(X_train, y_train)
    results_df, roc_curves = evaluate_models(models, X_test, y_test)
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]
    selected_features = feature_names
    top_features = selected_features[:5]  # 10 ویژگی برتر
else:
    models = {}
    results_df = pd.DataFrame()
    best_model = None
    selected_features = []
    top_features = []

# ============================================
# 3. ایجاد برنامه Dash
# ============================================

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# تولید فیلدهای فرم به صورت پویا
form_fields = []
for feat in top_features:
    if feat == 'Gender':
        field = dbc.Select(
            id=feat,
            options=[
                {"label": "زن", "value": 0},
                {"label": "مرد", "value": 1}
            ],
            value=0
        )
    elif feat in ['Hrct', 'Naghs-imeni', "Cough"]:
        field = dbc.Select(
            id=feat,
            options=[
                {"label": "ندارد", "value": 0},
                {"label": "دارد", "value": 1}
            ],
            value=0
        )
    else:
        field = dbc.Input(
            id=feat,
            type="number",
            min=0,
            max=120 if feat in ['Age','Temprature'] else None,
            step=0.1 if feat=='NUT1' else 1,
            placeholder=get_feature_description(feat)
        )

    form_fields.append(
        dbc.Row([
            dbc.Label(feat, width=4),
            dbc.Col([
                field,
                dbc.FormText(get_feature_description(feat))
            ], width=8)
        ], className="mb-3")
    )

app.layout = dbc.Container([
    # هدر
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="https://via.placeholder.com/30", height="30px")),
                    dbc.Col(dbc.NavbarBrand("داشبورد تشخیص احتمال ابتلا به بیماری کرونا", className="ms-2")),
                ], align="center", className="g-0"),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.Button("بروزرسانی داده", id="refresh-btn", color="light", className="me-2"),
            html.Span(id="last-update", className="text-white")
        ]),
        color="primary",
        dark=True
    ),

    # تب‌ها
    dbc.Tabs([
        # تب آمار توصیفی
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("آمار توصیفی داده‌ها"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                figure=px.histogram(raw_data, x='Age', nbins=20,
                                                    title='توزیع سن بیماران') if raw_data is not None else {}
                            )
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(
                                figure=px.box(raw_data, y='Temprature',
                                              title='دمای بدن بیماران') if raw_data is not None else {}
                            )
                        ], md=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                figure=px.imshow(raw_data[['Age', 'Temprature', 'NUT1']].corr(),
                                                 title='ماتریس همبستگی') if raw_data is not None else {}
                            )
                        ])
                    ])
                ])
            ], className="mt-4")
        ], label="📊 آمار توصیفی"),

        # تب تحلیل مدل
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("نتایج مدل‌های یادگیری ماشین"),
                dbc.CardBody([
                    dbc.Alert(
                        f"بهترین مدل: {best_model_name} با دقت {results_df.loc[results_df['Accuracy'].idxmax(), 'Accuracy']}" if best_model_name else "هیچ مدلی آموزش داده نشده",
                        color="success"
                    ),
                    dbc.Table.from_dataframe(
                        results_df.sort_values('Accuracy', ascending=False) if not results_df.empty else pd.DataFrame(),
                        striped=True,
                        bordered=True,
                        hover=True,
                        className="mb-4"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': k, 'value': k} for k in models.keys()],
                                value=[best_model_name] if best_model_name else None,
                                multi=True
                            ),
                            dcc.Graph(id='feature-importance-comparison')
                        ])
                    ])
                ])
            ], className="mt-4")
        ], label="🤖 تحلیل مدل"),

        # تب پیش‌بینی
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("پیش‌بینی نتیجه تست کرونا"),
                dbc.CardBody([
                    html.P("لطفاً اطلاعات بیمار را وارد نمایید:", className="lead"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Form(
                                form_fields + [
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "انجام پیش‌بینی",
                                                id="predict-btn",
                                                color="primary",
                                                className="w-100"
                                            ),
                                            dbc.Button(
                                                "بازنشانی فرم",
                                                id="reset-btn",
                                                color="secondary",
                                                className="w-100 mt-2"
                                            )
                                        ], width=4)
                                    ], justify="center", className="mt-4")
                                ]
                            )
                        ], md=8, className="mx-auto")
                    ]),

                    # نتیجه پیش‌بینی
                    dbc.Collapse(
                        dbc.Card(id="prediction-output", color="light", className="mt-4"),
                        id="prediction-collapse",
                        is_open=False
                    )
                ])
            ], className="mt-4")
        ], label="🔮 پیش‌بینی")
    ], className="mt-3"),

    # فوتر
    dbc.Row([
        dbc.Col([
            html.P("© 2025 سیستم پیش‌بینی کرونا - توسعه داده شده با Dash",
                   className="text-center text-muted mt-4")
        ])
    ])
], fluid=True)

st.set_page_config(page_title="پیش‌بینی با ویژگی های مهم", layout="centered")
st.title("🔮 پیش‌بینی نتیجه تست کرونا بر اساس ویژگی های مهم")

# تعریف 10 ویژگی مهم انتخاب شده از لایه دوم
top_10_features = [ "Hrct", "Naghs-imeni", "Palse_min", "Temprature", "NUT1", "O2sat"]

# بارگذاری داده از فایل یا منبع آنلاین (در صورت نیاز)
data_url = "https://raw.githubusercontent.com/mostafatamandi/covid_dashboard/main/clean_df.csv"
try:
    df = pd.read_csv(data_url)
    st.success("✅ داده‌ها با موفقیت بارگذاری شدند.")
except:
    st.error("❌ خطا در بارگذاری داده‌ها.")
    st.stop()

# حذف ستون‌های نامناسب (در صورت وجود)
drop_cols = ['PCR', 'Systole']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# حذف سطرهایی که دارای مقادیر گمشده هستند (برای سادگی)
df = df.dropna(subset=top_10_features + ['TEST'])

# انتخاب ویژگی‌ها و هدف
X = df[top_10_features].copy()
y = df['TEST']

# کدگذاری ویژگی‌های دسته‌ای
#if 'Gender' in X.columns and X['Gender'].dtype == 'object':
 #   X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})

if 'Naghs-imeni' in X.columns and X['Naghs-imeni'].dtype == 'object':
    X['Naghs-imeni'] = X['Naghs-imeni'].map({'NO': 0, 'YES': 1})

if 'Hrct' in X.columns and X['Hrct'].dtype == 'object':
    X['Hrct'] = X['Hrct'].map({'NO': 0, 'YES': 1})


# اگر سایر ویژگی‌ها هم دسته‌ای بودند، مشابه بالا کدگذاری شوند

# نمایش دندروگرام
st.subheader("📊 نمودار خوشه‌بندی سلسله‌مراتبی (Dendrogram)")
try:
    linkage_matrix = linkage(X, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=X.index, leaf_rotation=90)
    st.pyplot(fig)
except Exception as e:
    st.warning("نمودار دندروگرام قابل نمایش نیست.")
    st.text(str(e))

# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# آموزش مدل
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ذخیره مدل و Scaler
with open("best_model_top10.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler_top10.pkl", "wb") as f:
    pickle.dump(scaler, f)

st.subheader("🔧 فرم ورود داده‌ها برای پیش‌بینی:")
user_input = {}
for feature in top_10_features:
    user_input[feature] = st.number_input(f"مقدار {feature}", value=0.0, step=0.1)

# تبدیل ورودی کاربر به DataFrame و استانداردسازی
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# پیش‌بینی
if st.button("📈 پیش‌بینی کن"):
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0]

    if prediction[0] == 1:
        st.success("✅ نتیجه پیش‌بینی: فرد مبتلا به کرونا است")
        st.write(f"🔴 احتمال ابتلا: {proba[1]*100:.2f}%")
    else:
        st.success("✅ نتیجه پیش‌بینی: فرد مبتلا نیست")
        st.write(f"🟢 احتمال عدم ابتلا: {proba[0]*100:.2f}%")

# ============================================
# 4. Callback ها
# ============================================

@app.callback(
    [Output("last-update", "children"),
     Output("prediction-collapse", "is_open")],
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    return f"آخرین بروزرسانی: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", False


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(feat, "value") for feat in top_features],
    prevent_initial_call=True
)
def make_prediction(n_clicks, *input_values):
    if not best_model:
        return dbc.CardBody([
            html.H4("خطا در پیش‌بینی", style={'color': 'red', 'margin-bottom': '10px'}),
            html.P("مدل پیش‌بینی آماده نیست. لطفاً ابتدا داده‌ها را بروزرسانی کنید",
                   style={'font-size': '16px', 'color': '#555'})
        ])

    try:
        input_data = np.array(input_values).reshape(1, -1)
        prediction = best_model.predict(input_data)
        prediction_proba = best_model.predict_proba(input_data)[0]

        if prediction[0] == 1:
            return dbc.CardBody([
                html.H4("نتیجه پیش‌بینی: فردی با ویژگی های فوق کرونا دارد",
                        style={'color': 'red', 'margin-bottom': '10px'}),
                html.P(f"احتمال مثبت بودن: {prediction_proba[1] * 100:.2f}%",
                       style={'font-size': '16px', 'margin-bottom': '10px'}),
                html.P("توصیه: انجام آزمایش‌های تکمیلی و مشورت با پزشک",
                       style={'font-size': '16px', 'color': '#555'})
            ])
        else:
            return dbc.CardBody([
                html.H4("نتیجه پیش‌بینی: فردی با ویژگی های فوق کرونا ندارد",
                        style={'color': 'green', 'margin-bottom': '10px'}),
                html.P(f"احتمال منفی بودن: {prediction_proba[0] * 100:.2f}%",
                       style={'font-size': '16px', 'margin-bottom': '10px'}),
                html.P("توصیه: در صورت ادامه علائم، مجدداً آزمایش دهید",
                       style={'font-size': '16px', 'color': '#555'})
            ])
    except Exception as e:
        return dbc.CardBody([
            html.H4("خطا در پردازش", style={'color': 'red', 'margin-bottom': '10px'}),
            html.P("لطفاً مقادیر معتبر وارد کنید",
                   style={'font-size': '16px', 'color': '#555'}),
            html.P(str(e), style={'font-size': '14px', 'color': '#777'})
        ])


@app.callback(
    [Output(feat, "value") for feat in top_features],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_form(n_clicks):
    return [0 for _ in top_features]


@app.callback(
    Output('feature-importance-comparison', 'figure'),
    Input('model-selector', 'value')
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
                    name=model_name
                )
            )

    return {
        'data': data,
        'layout': go.Layout(
            title='مقایسه اهمیت متغیرها در مدل‌های انتخابی',
            xaxis_title='متغیرها',
            yaxis_title='میزان اهمیت',
            barmode='group',
            height=600
        )
    }


# ============================================
# 5. اجرای برنامه
# ============================================

if __name__ == '__main__':
    app.run(debug=True, port=8060)