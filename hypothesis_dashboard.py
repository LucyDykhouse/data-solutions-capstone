import pandas as pd
import pickle
import base64
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Retrieve data
df = pd.read_csv('data/ready_df.csv')
potable_df = df[df['potability'] == 1]
nonpotable_df = df[df['potability'] == 0]
print(df.head())

# Retrieve models
initial_model = pickle.load(open('models/logit_fit.sav', 'rb'))
final_model = pickle.load(open('models/logit_model.sav', 'rb'))

# Retrieve p-values from model
i = 0
pvalues = {}
while i < len(df.columns.tolist()) - 1:
    pvalues[df.columns.tolist()[i]] = [initial_model.pvalues.tolist()[i]]
    i += 1

pvalues = pd.DataFrame(pvalues)

# Retrieve coefficients from model
j = 0
coefs = {'ammonia': 0,
         'copper': 0,
         'fluoride': 0,
         'lead': 0,
         'radium': 0,
         'selenium': 0}

while j < len(df.drop(['ammonia', 'copper', 'fluoride',
                       'lead', 'radium', 'selenium'], axis=1).columns.tolist()) - 1:
    coefs[df.drop(['ammonia', 'copper', 'fluoride',
                   'lead', 'radium', 'selenium'], axis=1).columns.tolist()[j]] = [final_model.params.tolist()[j + 1]]
    j += 1

coefs = pd.DataFrame(coefs)

# Retrieve image that displays model
img_filename = 'models/log_reg_eqn.PNG'
encoding = base64.b64encode(open(img_filename, 'rb').read())

# Initialize dash app and styles
styles = 'https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cosmo/bootstrap.min.css'
app = dash.Dash(external_stylesheets=[styles])

# Define layout of app
app.layout = html.Div([
    html.Div([
        html.H1('Attribute Distributions', style={'text-align': 'center'}),
        dcc.Graph(id='distributions'),
        html.H1('Attribute P-values and Coefficients', style={'text-align': 'center', 'margin-top': '2rem', 'margin-bottom': '2rem'}),
        html.H3(
            'P-value meaning: how likely data is to have occurred under the null hypothesis (random chance). A p-value below the conventional level of significance (0.05) indicates that the effect of the explanatory variable on potability is statistically significant.'),
        html.H3(
            'Coefficient meaning: for each unit increase of the chosen explanatory variable, the log-odds of a water resource being classified as potable increase by the coefficient value.'),
        dcc.Graph(id='pvalues', style={'display': 'inline-block'}),
        dcc.Graph(id='coefs', style={'display': 'inline-block'}),
    ]),
    html.Div([
        html.H3('Select Attribute to Graph:'),
        dcc.Dropdown(id='options',
                     clearable=False,
                     options=[
                         {'label': attr, 'value': attr} for attr in df.iloc[:, 0:15].columns.tolist()
                     ],
                     value='ammonia'
                     )
    ]),
    html.Div([
        html.H1('Results of Hypothesis Testing', style={'text-align': 'center', 'margin-top': '3rem', 'margin-bottom': '2rem'}),
        html.H3(
            'Recall H_0: None of the 20 predictor variables have a statistically significant relationship with the response variable, water potability.'),
        html.H3(
            'Because not every coefficient is equal to 0, we have sufficient evidence to reject the null hypothesis and accept the alternative hypothesis, which states that there is a statistically significant relationship between the variables with a nonzero coefficient (and p-value < 0.05) and water potability.'),
        html.H3(
            'Given each coefficient and the attribute orders as determined in data wrangling, the potability of water can be modeled as follows, where the left side of the equation represents the log odds of potability.'),
        html.Img(src='data:image/png;base64,{}'.format(encoding.decode()), style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
    ]),
    html.Div([
        html.H1('Model Predictions', style={'text-align': 'center', 'margin-top': '3rem', 'margin-bottom': '2rem'}),
        html.H3(
            'Log-odds are not intuitive, so prediction is used to measure the accuracy, precision, and recall of the regression/classification.'),
        dcc.Graph(id='matrix'),
        html.H3('Accuracy = proportion of correct predictions = (1630) / (1865) = 0.874'),
        html.H3('Precision = how well model performs when the prediction is positive = (14) / (35) = 0.400'),
        html.H3('Recall = how well model predicts positives = (14) / (228) = 0.061', style={'margin-bottom': '2rem'}),
        html.H3(
            'While the model is quite accurate, it has very poor precision and recall. In the case of water potability, precision is more of an issue in that false positives are much worse than false negatives.'),
        html.H3(
            'One reason why the model has such poor precision and recall is the imbalance of the data - about 15% of the training and testing dataset values were potable. Oversampling may increase both the model\'s tendency to predict positives and its accuracy when doing so.')
    ])
], style={'padding': '2rem',
          'margin': '2rem'})


# Get app inputs and outputs
@app.callback([
    Output('distributions', 'figure'),
    Output('pvalues', 'figure'),
    Output('coefs', 'figure'),
    Output('matrix', 'figure')
],
    [Input('options', 'value')]
)
# Create app figures
def update_graphs(attribute):
    """ Function for generating the histogram, p-value, and coefficient of the chosen explanatory variable, along with the confusion matrix.

    :param attribute: Explanatory variable chosen from dropdown
    :return: [fig1, fig2, fig3, fig4]: List of figures generated in the function
    """

    # Fig1 = distribution of chosen attribute
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=nonpotable_df[attribute], name='Nonpotable'))
    fig1.add_trace(go.Histogram(x=potable_df[attribute], name='Potable'))

    fig1.update_layout(title_text='Distribution of Water Attributes',
                       title_x=0.5,
                       xaxis_title='{} values ^ order (ppm/CFU ^ order)'.format(attribute),
                       yaxis_title='Count')

    # Fig2 = P-value gauge
    fig2 = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pvalues.loc[0, attribute].round(3),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'P-value of {}'.format(attribute)},
        gauge={'axis': {'range': [None, 1.00]},
               'bar': {'color': 'darkblue'},
               'threshold': {
                   'line': {'color': 'orange', 'width': 3},
                   'thickness': 0.75, 'value': 0.05}}
    ))

    # Fig3 = Coefficient Gauge
    fig3 = go.Figure(go.Indicator(
        mode='gauge+number',
        value=coefs.loc[0, attribute].round(3),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Coefficient of {}'.format(attribute)},
        gauge={'axis': {'range': [-50.0, 50.0]},
               'bar': {'color': 'darkblue'}}
    ))

    # Fig4 = Confusion matrix
    fig4 = go.Figure(data=go.Heatmap(
        x=['nonpotable', 'potable'],
        y=['potable  ', 'nonpotable  '],
        z=[[214, 14],
           [1616, 21]]
    ))

    fig4.update_layout(title_text='Confusion Matrix',
                       title_x=0.5,
                       yaxis_title='Observed',
                       xaxis_title='Predicted')

    return [fig1, fig2, fig3, fig4]


# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
