import pandas as pd
import panel as pn
import plotly.graph_objs as go
import holoviews as hv
from holoviews import opts, dim
import param
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')
df ["average score"] = (df ['math score'] + df['reading score'] + df['writing score']) /3 

class ML_dashboard (param.Parameterized): 

    culter_widget = param.Integer(3, bounds=(1, 10))

    @param.depends('culter_widget',watch=True, on_init=False)
    def k_means (self):
        # Load data
        data = df

        # Convert categorical variables to numerical variables
        data = pd.get_dummies(data, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

        # Select columns for k-means clustering
        X = data[['math score', 'reading score', 'writing score', 'average score']]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.culter_widget, random_state=42)
        kmeans.fit(X_scaled)

        # Add cluster labels to data
        data['cluster'] = kmeans.labels_ 

        # Create 2D scatter plot
        #fig1 = px.scatter(data, x='math score', y='reading score', color='cluster')
        # Create trace for each cluster
        traces1 = []
        for cluster in data['cluster'].unique():
            cluster_data = data[data['cluster'] == cluster]
            trace = go.Scatter(
                x=cluster_data['math score'],
                y=cluster_data['reading score'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=8,
                    color=cluster,
                    colorscale='sunset'
                )
            )
            traces1.append(trace)

        # Create layout
        layout1 = go.Layout(
            title='K-Means Clustering',
            xaxis=dict(title='Math Score'),
            yaxis=dict(title='Reading Score')
        )

        # Create figure
        fig1 = go.Figure(data=traces1, layout=layout1) 
        return  pn.Row(
            pn.Column(fig1, height=600, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both') 
    
    @param.depends('culter_widget',watch=True, on_init=False)
    def k_means_3d (self):
        # Load data
        data = df

        # Convert categorical variables to numerical variables
        data = pd.get_dummies(data, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

        # Select columns for k-means clustering
        X = data[['math score', 'reading score', 'writing score', 'average score']]
       
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.culter_widget, random_state=42)
        kmeans.fit(X_scaled)

        # Add cluster labels to data
        data['cluster'] = kmeans.labels_ 

        # Create 3D scatter plot
        #fig = px.scatter_3d(data, x='math score', y='reading score', z='writing score', color='cluster') 

        # Create trace for each cluster
        traces = []
        for cluster in data['cluster'].unique():
            cluster_data = data[data['cluster'] == cluster]
            trace = go.Scatter3d(
                x=cluster_data['math score'],
                y=cluster_data['reading score'],
                z=cluster_data['writing score'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=8,
                    color=cluster,
                    colorscale='matter'
                )
            )
            traces.append(trace)

        # Create layout
        layout = go.Layout(
            title='K-Means Clustering 3D',
            scene=dict(
                xaxis=dict(title='Math Score'),
                yaxis=dict(title='Reading Score'),
                zaxis=dict(title='Writing Score')
            )
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        return  pn.Row(
            pn.Column(fig, height=600, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both') 
    
    def PCA (self):
        # Load data
        data = df 
        # Perform PCA with 2 components
        pca = PCA(n_components=2).fit(data[['math score', 'reading score', 'writing score']])
        transformed = pca.transform(data[['math score', 'reading score', 'writing score']])
        data['pca_x'] = transformed[:, 0]
        data['pca_y'] = transformed[:, 1]

        # Create a scatter plot of the PCA results
        fig = go.Figure()
        for label in data['race/ethnicity'].unique():
            fig.add_trace(go.Scatter(
                x=data.loc[data['race/ethnicity']==label, 'pca_x'],
                y=data.loc[data['race/ethnicity']==label, 'pca_y'],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=data.loc[data['race/ethnicity']==label, 'average score'],
                    colorscale='sunsetdark'
                )
            ))

        # Set the layout
        fig.update_layout(
            title='PCA Results',
            xaxis=dict(title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)'),
            yaxis=dict(title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'),
        )

        return  pn.Row(
            pn.Column(fig, height=600, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both') 
   
    def PCA_3d (self):
        # Load data
        data = df 
        # Perform PCA with 3 components
        pca = PCA(n_components=3).fit(data[['math score', 'reading score', 'writing score']])
        transformed = pca.transform(data[['math score', 'reading score', 'writing score']])
        data['pca_x'] = transformed[:, 0]
        data['pca_y'] = transformed[:, 1]
        data['pca_z'] = transformed[:, 2]

        # Create a 3D scatter plot of the PCA results
        fig = go.Figure()
        for label in data['race/ethnicity'].unique():
            fig.add_trace(go.Scatter3d(
                x=data.loc[data['race/ethnicity']==label, 'pca_x'],
                y=data.loc[data['race/ethnicity']==label, 'pca_y'],
                z=data.loc[data['race/ethnicity']==label, 'pca_z'],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=data.loc[data['race/ethnicity']==label, 'average score'],
                    colorscale='sunsetdark'
                )
            ))

        # Set the layout
        fig.update_layout(
            title='PCA Results',
            scene=dict(
                xaxis=dict(title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)'),
                yaxis=dict(title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'),
                zaxis=dict(title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
            )
        )
        return  pn.Row(
            pn.Column(fig, height=600, sizing_mode='stretch_both', width_policy='max'),
            sizing_mode='stretch_both') 
   
   
dashboard = ML_dashboard()
# Define a custom panel template with the grid layout
template = pn.template.FastListTemplate(
    title='Student dashboard',
    main=[dashboard.k_means,dashboard.k_means_3d,dashboard.PCA,dashboard.PCA_3d],
    sidebar_width=305,
    header_background = "#A01346 ",
    theme_toggle = False,
    logo  = "ML.ico",

     sidebar=[pn.Param(dashboard.param,widgets={'math_widget':pn.widgets.FloatSlider,'reading_widget':pn.widgets.FloatSlider,'writing_widget':pn.widgets.FloatSlider, 'race_ethnicity_widget': pn.widgets.Select,'gender_widget': pn.widgets.Select, 'parental_level_of_education_widget': pn.widgets.Select, 'lunch_widget': pn.widgets.Select, 'test_preparation_course_widget': pn.widgets.Select})]
)

# Show the template in a browser tab
template.servable()