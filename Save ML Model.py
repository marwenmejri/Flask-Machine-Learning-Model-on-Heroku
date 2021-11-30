from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import wget
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from joblib import dump, load

pio.renderers.default = "svg"
# url = "https://archive.org/download/ages-and-heights/AgesAndHeights.pkl"
# filename = wget.download(url)

# data = pd.read_pickle(filename)

data = pd.read_csv("AgesAndHeights.csv")
data = data[data["Age"] >0]

ages = data["Age"]
ages.hist()

heights = data["Height"]
heights.hist()

data.plot.scatter(x="Age", y="Height")

fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", 
                 labels={'x':"Age (Years)", 'y':"Height (inches)"})
fig.show()

ages_np = ages.to_numpy()
heights_np = heights.to_numpy()
ages_np.shape
ages_np_reshaped = ages_np.reshape(len(ages_np), 1)
print(ages_np_reshaped)

model = LinearRegression().fit(X=ages_np_reshaped, y=heights_np, )

print(model.intercept_)
print(model.coef_)

x_new = np.array(list(range(19))).reshape(19,1)
print(x_new)
preds = model.predict(X=x_new)

fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", 
                 labels={'x':"Age (Years)", 'y':"Height (inches)"})
fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode="lines", name="Model"))
fig.show()
fig.write_image("base_pic.svg", width=800)


dump(model, 'model.joblib')

model_in = load('model.joblib')
model_in.predict(np.array([[1]]))


def floats_string_to_np_array(float_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in float_str.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)


floats_string_to_np_array("1, 2,    5,    17.68")


def make_picture(training_data_filename, model, new_inp_np_arr, output_file='predictions_pic.svg'):
    data = pd.read_csv(training_data_filename)
    data = data[data["Age"] >0]
    ages = data["Age"]
    heights = data["Height"]
    x_new = np.array(list(range(19))).reshape(19,1)
    preds = model.predict(X=x_new)
    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", 
                     labels={'x':"Age (Years)", 'y':"Height (inches)"})
    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode="lines", name="Model"))
    new_preds = model.predict(new_inp_np_arr)

    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='yellow', size=20, line=dict(color='purple', width=2))))
    fig.show()
    fig.write_image(output_file, width=800)
    
    return fig
    
make_picture("AgesAndHeights.csv", model_in, floats_string_to_np_array("1, 2,    5,    17.68"), output_file='base_pic.svg' )  















