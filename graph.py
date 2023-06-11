#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import plotly
# from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import colorlover as cl
import numpy as np

import sys

# reload(sys)
# sys.setdefaultencoding('utf8')


########################################################################################################################
class visualisation(object):
    """
        Visualisation class using the plot.ly platform. This class only defines the template to make the plots.
    """
    def __init__(self, renderer=""):
        """
            define plotly renderer - use pio.renderers to get list of renderers
        """
        pio.renderers.default = renderer

    # ----------------------------------------------------------------------------------------------------------------------

    def layout_create(self):
        return go.Layout()

    def layout(self, title='', x_label='', y_label='', xrange=[], yrange=[], theme='', height=0, width=0):
        l = self.layout_create()
        l = self.layout_update(l, title=title, x_label=x_label, y_label=y_label, xrange=xrange, yrange=yrange, theme=theme, height=height, width=width)
        return l

    def layout_update(self, l, title='', x_label='', y_label='', xrange=[], yrange=[], theme='', height=0, width=0):
        if theme == '':
            theme = 'light'

        l.update(
            title={'text': title, 'font': {'size': 16, 'color': '#AAAAAA'}},
            autosize=True,
            margin=go.layout.Margin(l=60, r=60, b=60, t=60, pad=0),
            xaxis=dict(
                title=x_label,
                titlefont=dict( size=12 )
            ),
            yaxis=dict(
                title=y_label,
                titlefont=dict( size=12 )
            ),
            scene=dict(
                xaxis=dict(
                    title='',
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                yaxis=dict(
                    title='',
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                zaxis=dict(
                    title='',
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.7, y=0.7, z=0.7)
                )
            )
        )
        
        # Update axis ranges
        l.xaxis.range = xrange if xrange != [] else None                
        l.yaxis.range = yrange if yrange != [] else None

        # Update size
        l.update(height=height) if height > 0 else None
        l.update(width=width) if width>0 else None
        
        if theme == 'dark':
            l.update(paper_bgcolor='#222222',
                     plot_bgcolor='#222222',
                 xaxis={'color': '#AAAAAA',
                        'tickfont': {'size': 16},
                        'gridcolor': '#555555',
                        'zerolinecolor': '#555555',
                        'titlefont': {'color': '#AAAAAA', 'size': 16}},
                 yaxis={'color': '#AAAAAA',
                        'tickfont': {'size': 16},
                        'gridcolor': '#555555',
                        'zerolinecolor': '#555555',
                        'titlefont': {'color': '#AAAAAA', 'size': 16}},
                     legend={'font': {'color': '#AAAAAA'}})

        if theme == 'gray':
            l.update(title="",
                     plot_bgcolor='#DDDDDD',
                     xaxis={'color': '#555555', 'titlefont': {'color': '#555555', 'size': 18},
                            'tickfont': {'size': 18}},
                     yaxis={'color': '#555555', 'titlefont': {'color': '#555555', 'size': 18},
                            'tickfont': {'size': 18}},
                     legend={'font': {'color': '#222222'}})

        if theme == 'light':
            l.update(paper_bgcolor='#FFFFFF',
                     plot_bgcolor='#EEEEEE',
                     xaxis={'color': '#555555', 'titlefont': {'color': '#555555'}},
                     yaxis={'color': '#555555', 'titlefont': {'color': '#555555'}},
                     legend={'font': {'color': '#222222'}})

        return l

    # -------------------------------------------------------------------------------------------------------------------
    def bar(self, x, y, name='', color=0, opacity=0.8):
        return go.Bar(
            x = x,
            y = y,
            name = name,
            marker = dict(
                color = self.color(color, opacity), # 'rgba(255, 50, 50, 0.9)', # 'rgb(158,202,225)'
                line = dict(color=self.color(color, opacity), width=1), # 'rgb(8,48,107)'
            ),
            opacity = opacity
        )
    # -------------------------------------------------------------------------------------------------------------------
    def plot(self, x, y, name='', color=0, opacity=0.2, showlegend=True, fill='tozeroy', linewidth=1):
        return go.Scatter(
        name=name,
        x=x,
        y=y,
        fill=fill,
        fillcolor = self.color(color, opacity),
        marker=dict(
            color=self.color(color, opacity+0.3), # 'rgba(150, 100, 100, 0.7)',
            line=dict(color=self.color(color, opacity), width=linewidth),
            symbol='circle',
            size=10,
            opacity=opacity
        ),
        showlegend=showlegend
    )

    # -------------------------------------------------------------------------------------------------------------------
    def scatter(self, x, y, text=None, name='', color=0, opacity=0.4, size=10):
        return go.Scatter(
            name=name,
            x=x,
            y=y,
            text=text,
            mode='markers',
            marker=dict(
                color=self.color(color, opacity+0.3), #'rgba(255, 100, 100, 0.9)',
                #                 line=dict(color=self.color(color, opacity), width=1), # 'rgba(150, 50, 50, 0.9)',
                symbol='circle',
                size=size,
                opacity=opacity
            )
        )
    # -------------------------------------------------------------------------------------------------------------------
    def scatter3d(self, x, y, z, name='', color=0, opacity=0.8):
        return go.Scatter3d(
            name=name,
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker=dict(
                color=self.color(color, opacity+0.3), #'rgba(255, 100, 100, 0.9)',
                #                 line=dict(color=self.color(color, opacity), width=1), # 'rgba(150, 50, 50, 0.9)',
                symbol='circle',
                size=10,
                opacity=opacity
            )
        )

    # -------------------------------------------------------------------------------------------------------------------
    def box(self, x, y, name='', color=0, opacity=0.8):
        return go.Box(
            x = x,
            y = y,
            name = name,
            marker = dict(
                color = self.color(color, opacity), # 'rgba(255, 50, 50, 0.9)', # 'rgb(158,202,225)'
                line = dict(color=self.color(color, opacity), width=1), # 'rgb(8,48,107)'
            ),
            opacity = opacity
        )
    # -------------------------------------------------------------------------------------------------------------------
    def color(self, c, opacity=1.0):
        cl.scales['custom'] = ['rgba(250, 100, 100, {})'.format(opacity),
                               'rgba(150, 100, 150, {})'.format(opacity),
                               'rgba(150, 50, 50, {})'.format(opacity),
                              ] \
                            + ["rgba" + _[3:-1] + ",{})".format(opacity)for _ in cl.scales['11']['div']['RdGy']]
        n_colors = len(cl.scales['custom'])
        return c if type(c) == str else cl.scales['custom'][np.mod(c,n_colors)]




########################################################################################################################
if __name__ == '__main__':

    x = range(1, 10)
    y = [4, 5, 7, 2, 5, 7, 1, 6, 8, 10]

    # Init the thing
    plot = visualisation()

    # Make the plot as an html file, shown separately
    data = plot.bar(x=x, y=y)
    # data.hovertext = foobar
    layout = plot.layout(title='test', x_label='my x', y_label='my y')
    fig = go.Figure(data=[data], layout=layout)
    py.plot(fig, filename='goat')
