import pandas as pd
import panel as pn
import plotly.express as px
from sklearn.linear_model import LinearRegression
''' 
# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# Define the sidebar widgets
math_score_filter = pn.widgets.FloatSlider(name='Math score', start=df['math score'].min(), end=df['math score'].max(), value=df['math score'].mean())
reading_score_filter = pn.widgets.FloatSlider(name='Reading score', start=df['reading score'].min(), end=df['reading score'].max(), value=df['reading score'].mean())
writing_score_filter = pn.widgets.FloatSlider(name='Writing score', start=df['writing score'].min(), end=df['writing score'].max(), value=df['writing score'].mean())

# Define the function for linear regression
def linear_regression(df, x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    coef = reg.coef_[0]
    intercept = reg.intercept_
    return score, coef, intercept

# Define the function for updating the plot
def update_plot(math_score, reading_score, writing_score):
    filtered_df = df[(df['math score'] >= math_score) & (df['reading score'] >= reading_score) & (df['writing score'] >= writing_score)]
    score1, coef1, intercept1 = linear_regression(filtered_df, 'math score', 'reading score')
    score2, coef2, intercept2 = linear_regression(filtered_df, 'math score', 'writing score')
    fig = px.scatter(filtered_df, x='math score', y='reading score', color='gender', title='Math vs. Reading Scores')
    fig.add_trace(px.line(x=[filtered_df['math score'].min(), filtered_df['math score'].max()], y=[coef1*filtered_df['math score'].min()+intercept1, coef1*filtered_df['math score'].max()+intercept1], name='Reading Score').data[0])
    fig.add_trace(px.line(x=[filtered_df['math score'].min(), filtered_df['math score'].max()], y=[coef2*filtered_df['math score'].min()+intercept2, coef2*filtered_df['math score'].max()+intercept2], name='Writing Score').data[0])
    return fig

# Define the main panel
sidebar = pn.Column(math_score_filter, reading_score_filter, writing_score_filter)
plot = pn.pane.Plotly(height=400)

dashboard = pn.Row(sidebar, plot)

# Update the plot when the filters are changed
@pn.depends(math_score_filter.param.value, reading_score_filter.param.value, writing_score_filter.param.value)
def update_dashboard(math_score, reading_score, writing_score):
    plot.object = update_plot(math_score, reading_score, writing_score)

# Run the app
dashboard.servable()

'''
import pandas as pd
import panel as pn

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# Define the sidebar widgets
num_rows = 15
num_rows_input = pn.widgets.IntSlider(name='Number of rows', start=1, end=len(df), value=num_rows)
gender_filter = pn.widgets.Select(name='Gender', options=['All'] + df['gender'].unique().tolist(), value='All')
ethnicity_filter = pn.widgets.Select(name='Ethnicity', options=['All'] + df['race/ethnicity'].unique().tolist(), value='All')
parent_ed_filter = pn.widgets.Select(name='Parental education', options=['All'] + df['parental level of education'].unique().tolist(), value='All')

# Define the function for updating the table
def update_table(num_rows, gender, ethnicity, parent_ed):
    filtered_df = df
    if gender != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    if ethnicity != 'All':
        filtered_df = filtered_df[filtered_df['race/ethnicity'] == ethnicity]
    if parent_ed != 'All':
        filtered_df = filtered_df[filtered_df['parental level of education'] == parent_ed]
    filtered_df = filtered_df.head(num_rows)
    return filtered_df

# Define the main panel
table = pn.widgets.DataFrame(update_table(num_rows, gender_filter.value, ethnicity_filter.value, parent_ed_filter.value), width=800)

sidebar = pn.Column(num_rows_input, gender_filter, ethnicity_filter, parent_ed_filter, pn.layout.HSpacer(), pn.widgets.Button(name='Refresh', button_type='primary'))

dashboard = pn.Row(sidebar, table)

# Update the table when the filters are changed
@pn.depends(num_rows_input.param.value, gender_filter.param.value, ethnicity_filter.param.value, parent_ed_filter.param.value)
def update_dashboard(num_rows, gender, ethnicity, parent_ed):
    table.value = update_table(num_rows, gender, ethnicity, parent_ed)

# Run the app
dashboard.servable() 
