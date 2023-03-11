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
import hvplot.pandas
import plotly.graph_objs as go
import param

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')


class student_EDA (param.Parameterized): 


    gender_list= (list)(df["gender"].unique())
    gender_list.append('ALL')
    gender_widget  =param.ObjectSelector(default='ALL',objects=gender_list,label="Gender")

    race_ethnicity_list= (list)(df["race/ethnicity"].unique())
    race_ethnicity_list.append('ALL')
    race_ethnicity_widget  = param.ObjectSelector(default='ALL',objects=race_ethnicity_list,label="Race/Ethnicity")

    parental_level_of_education_list = (list)(df["parental level of education"].unique())
    parental_level_of_education_list.append('ALL')
    parental_level_of_education_widget  = param.ObjectSelector(default='ALL',objects=parental_level_of_education_list,label="Parental level of education")
    
    lunch_list= (list)(df["lunch"].unique())
    lunch_list.append('ALL')
    lunch_widget = param.ObjectSelector(default='ALL',objects=lunch_list,label="Lunch")

    test_prep_list= (list)(df["test preparation course"].unique())
    test_prep_list.append('ALL')
    test_preparation_course_widget  = param.ObjectSelector(default='ALL',objects=test_prep_list,label="Test preparation course")
    
    math_widget = param.Number(0, bounds=(0, 100))
    reading_widget  = param.Number(0, bounds=(0, 100))
    writing_widget  = param.Number(0, bounds=(0, 100))



    @pn.depends('gender_widget','race_ethnicity_widget','parental_level_of_education_widget','lunch_widget','test_preparation_course_widget','math_widget','reading_widget','writing_widget',watch=True, on_init=False)
    def table (self):
         # Define the filters dictionary and filter_table widget
        filters = {
        'gender': {'type': 'input', 'func': 'like', 'placeholder': 'Enter gender'},
        'race/ethnicity': {'type': 'input','placeholder': 'Enter race'},
        'parental level of education': {'type': 'input', 'func': 'like', 'placeholder': 'Enter parental level of education'},
        'lunch': {'type': 'input', 'func': 'like', 'placeholder': 'Enter director'},
        'test preparation course': {'type': 'input', 'func': 'like', 'placeholder': 'Enter writer'},
        'reading score': {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum reading score'},
        'math score': {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum math score'},
        'writing score': {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum rating'}
         }
        df1=df
        if (self.gender_widget!='ALL'):
            df1 = df1[df1['gender']==self.gender_widget]

        if (self.race_ethnicity_widget!='ALL'):
            df1 = df1[df1['race/ethnicity']==self.race_ethnicity_widget]

        if (self.parental_level_of_education_widget!='ALL'):
            df1 = df1[df1["parental level of education"]==self.parental_level_of_education_widget]

        if (self.lunch_widget!='ALL'):
            df1 = df1[df1['lunch']==self.lunch_widget]

        if (self.test_preparation_course_widget!='ALL'):
            df1 = df1[df1['test preparation course']==self.test_preparation_course_widget]
        df1 = df1[df1['math score']>=self.math_widget]
        df1 = df1[df1['reading score']>=self.reading_widget]
        df1 = df1[df1['writing score']>=self.writing_widget]


        filter_table =pn.widgets.Tabulator(df1, pagination='remote', layout='fit_columns', page_size=10, sizing_mode='stretch_width', header_filters=filters)
        return  pn.Row(
            pn.Spacer(width=10),  # Add some spacing between the sidebar and the filter_table
            pn.Column(filter_table, height=500, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both'
        )  
    @pn.depends('gender_widget','race_ethnicity_widget','parental_level_of_education_widget','lunch_widget','test_preparation_course_widget','math_widget','reading_widget','writing_widget',watch=True, on_init=False)
    def plots(self):
        df1=df
        if (self.gender_widget!='ALL'):
            df1 = df1[df1['gender']==self.gender_widget]

        if (self.race_ethnicity_widget!='ALL'):
            df1 = df1[df1['race/ethnicity']==self.race_ethnicity_widget]

        if (self.parental_level_of_education_widget!='ALL'):
            df1 = df1[df1["parental level of education"]==self.parental_level_of_education_widget]

        if (self.lunch_widget!='ALL'):
            df1 = df1[df1['lunch']==self.lunch_widget]

        if (self.test_preparation_course_widget!='ALL'):
            df1 = df1[df1['test preparation course']==self.test_preparation_course_widget]
            
        df1 = df1[df1['math score']>=self.math_widget]
        df1 = df1[df1['reading score']>=self.reading_widget]
        df1 = df1[df1['writing score']>=self.writing_widget]
        
        # Define the scatter plot and histogram
        scatter_plot = df1.hvplot.scatter(x='math score', y='reading score', c='writing score', cmap='viridis')
        histogram = df1.hvplot.hist('writing score', bins=20, width=500)
        # Bar chart
        bar_chart_data = df1.groupby('gender').mean()[['math score', 'reading score', 'writing score']]
        bar_chart = go.Figure(
            data=[
                go.Bar(name='Math score', x=bar_chart_data.index, y=bar_chart_data['math score']),
                go.Bar(name='Reading score', x=bar_chart_data.index, y=bar_chart_data['reading score']),
                go.Bar(name='Writing score', x=bar_chart_data.index, y=bar_chart_data['writing score']),
            ],
            layout=go.Layout(title='Average Scores by Gender')
        )
        bar_chart_panel = pn.pane.Plotly(bar_chart, sizing_mode='stretch_both')

        # Pie chart
        pie_chart_data = df1['race/ethnicity'].value_counts()
        pie_chart = go.Figure(
            data=[go.Pie(labels=pie_chart_data.index, values=pie_chart_data.values)],
            layout=go.Layout(title='Distribution of Students by Race/Ethnicity')
        )
        pie_chart_panel = pn.pane.Plotly(pie_chart, sizing_mode='stretch_both')

        # Combine the histogram and scatter_plot into a grid layout
        grid1 = pn.Row(
            pn.Column(histogram, height=500, sizing_mode='stretch_both', width_policy='max'),
            pn.Spacer(width=20),  # Add some spacing between the sidebar and the filter_table
            pn.Column(scatter_plot, height=500, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both'
        )

        # Combine the new plots into a grid layout with the existing ones
        grid2 = pn.Row(
            pn.Column(bar_chart_panel, height=500, sizing_mode='stretch_both', width_policy='max'),
            pn.Spacer(width=20),
            pn.Column(pie_chart_panel, height=500, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both'
        )
        return grid1
    
    @pn.depends('gender_widget','race_ethnicity_widget','parental_level_of_education_widget','lunch_widget','test_preparation_course_widget','math_widget','reading_widget','writing_widget',watch=True, on_init=False)
    def plots2(self):
        df1=df
        if (self.gender_widget!='ALL'):
            df1 = df1[df1['gender']==self.gender_widget]

        if (self.race_ethnicity_widget!='ALL'):
            df1 = df1[df1['race/ethnicity']==self.race_ethnicity_widget]

        if (self.parental_level_of_education_widget!='ALL'):
            df1 = df1[df1["parental level of education"]==self.parental_level_of_education_widget]

        if (self.lunch_widget!='ALL'):
            df1 = df1[df1['lunch']==self.lunch_widget]

        if (self.test_preparation_course_widget!='ALL'):
            df1 = df1[df1['test preparation course']==self.test_preparation_course_widget]
            
        df1 = df1[df1['math score']>=self.math_widget]
        df1 = df1[df1['reading score']>=self.reading_widget]
        df1 = df1[df1['writing score']>=self.writing_widget]
        
        # Bar chart
        bar_chart_data = df1.groupby('gender').mean()[['math score', 'reading score', 'writing score']]
        bar_chart = go.Figure(
            data=[
                go.Bar(name='Math score', x=bar_chart_data.index, y=bar_chart_data['math score']),
                go.Bar(name='Reading score', x=bar_chart_data.index, y=bar_chart_data['reading score']),
                go.Bar(name='Writing score', x=bar_chart_data.index, y=bar_chart_data['writing score']),
            ],
            layout=go.Layout(title='Average Scores by Gender')
        )
        bar_chart_panel = pn.pane.Plotly(bar_chart, sizing_mode='stretch_both')

        # Pie chart
        pie_chart_data = df1['race/ethnicity'].value_counts()
        pie_chart = go.Figure(
            data=[go.Pie(labels=pie_chart_data.index, values=pie_chart_data.values)],
            layout=go.Layout(title='Distribution of Students by Race/Ethnicity')
        )
        pie_chart_panel = pn.pane.Plotly(pie_chart, sizing_mode='stretch_both')

        # Combine the new plots into a grid layout with the existing ones
        grid2 = pn.Row(
            pn.Column(bar_chart_panel, height=500, sizing_mode='stretch_both', width_policy='max'),
            pn.Spacer(width=20),
            pn.Column(pie_chart_panel, height=500, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both'
        )
        return grid2
    
dashboard = student_EDA()
# Define a custom panel template with the grid layout
template = pn.template.FastListTemplate(
    title='Filter Table Template',
    main=[dashboard.table,dashboard.plots,dashboard.plots2],
     sidebar=[pn.Param(dashboard.param,widgets={'math_widget':pn.widgets.FloatSlider,'reading_widget':pn.widgets.FloatSlider,'writing_widget':pn.widgets.FloatSlider, 'race_ethnicity_widget': pn.widgets.Select,'gender_widget': pn.widgets.Select, 'parental_level_of_education_widget': pn.widgets.Select, 'lunch_widget': pn.widgets.Select, 'test_preparation_course_widget': pn.widgets.Select})]
)

# Show the template in a browser tab
template.servable()