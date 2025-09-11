#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import plotly
# from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import colorlover as cl
import numpy as np

# Import for video creation uisng matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.colors import rgb2hex


__author__ = "Morten Arngren"


########################################################################################################################
class Visualisation(object):
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

    def layout(self, title="", x_label="", y_label="", xrange=[], yrange=[], theme="", height=0, width=0):
        l = self.layout_create()
        l = self.layout_update(
            l,
            title=title,
            x_label=x_label,
            y_label=y_label,
            xrange=xrange,
            yrange=yrange,
            theme=theme,
            height=height,
            width=width,
        )
        return l

    def layout_update(self, l, title="", x_label="", y_label="", xrange=[], yrange=[], theme="", height=0, width=0):
        if theme == "":
            theme = "light"

        l.update(
            title={"text": title, "font": {"size": 16, "color": "#AAAAAA"}},
            autosize=True,
            margin=go.layout.Margin(l=60, r=60, b=60, t=60, pad=0),
            xaxis={'title': {'text': x_label, 'font': {'size': 14}}},
            yaxis={'title': {'text': y_label, 'font': {'size': 14}}},
            scene=dict(
                xaxis=dict(
                    title="",
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                zaxis=dict(
                    title="",
                    showgrid=False,
                    zeroline=True,
                    showline=False,
                    showticklabels=False,
                    showbackground=False,
                ),
                camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.7, y=0.7, z=0.7)),
            ),
        )

        # Update axis ranges
        l.xaxis.range = xrange if xrange != [] else None
        l.yaxis.range = yrange if yrange != [] else None

        # Update size
        l.update(height=height) if height > 0 else None
        l.update(width=width) if width > 0 else None

        if theme == "dark":
            l.update(
                title={'text': title, 'font': {'size': 16, 'color': '#AAAAAA'}},
                paper_bgcolor="#111122",
                plot_bgcolor="#111122",
                xaxis={
                    "color": "#AAAAAA",
                    "gridcolor": "#555555",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#AAAAAA"}},
                },
                yaxis={
                    "color": "#AAAAAA",
                    "gridcolor": "#555555",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#AAAAAA"}},
                },
                legend={"font": {"color": "#AAAAAA"}},
            )

        if theme == "gray":
            l.update(
                title={'text': title, 'font': {'size': 16, 'color': '#555555'}},
                paper_bgcolor="#DDDDDD",
                plot_bgcolor="#DDDDDD",
                xaxis={
                    "color": "#555555",
                    "gridcolor": "#BBBBBB",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#555555"}},
                },
                yaxis={
                    "color": "#555555",
                    "gridcolor": "#BBBBBB",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#555555"}},
                },
                legend={"font": {"color": "#222222"}},
            )

        if theme == "light":
            l.update(
                title={'text': title, 'font': {'size': 16, 'color': '#555555'}},
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#EEEEEE",
                xaxis={
                    "color": "#555555",
                    "gridcolor": "#BBBBBB",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#555555"}},
                },
                yaxis={
                    "color": "#555555",
                    "gridcolor": "#BBBBBB",
                    "zerolinecolor": "#555555",
                    "tickfont": {"size": 14},
                    "title": {"font": {"size": 14, "color": "#555555"}},
                },
                legend={"font": {"color": "#222222"}},
            )

        return l

    # -------------------------------------------------------------------------------------------------------------------
    def bar(self, x, y, name="", color=0, opacity=0.8):
        return go.Bar(
            x=x,
            y=y,
            name=name,
            marker=dict(
                color=self.color(color, opacity),  # 'rgba(255, 50, 50, 0.9)', # 'rgb(158,202,225)'
                line=dict(color=self.color(color, opacity), width=1),  # 'rgb(8,48,107)'
            ),
            opacity=opacity,
        )

    # -------------------------------------------------------------------------------------------------------------------
    def plot(self, x, y, name="", color=0, opacity=0.2, showlegend=True, fill="tozeroy", linewidth=1):
        return go.Scatter(
            name=name,
            x=x,
            y=y,
            fill=fill,
            fillcolor=self.color(color, opacity),
            marker=dict(
                color=self.color(color, opacity + 0.3),  # 'rgba(150, 100, 100, 0.7)',
                line=dict(color=self.color(color, opacity), width=linewidth),
                symbol="circle",
                size=10,
                opacity=opacity,
            ),
            showlegend=showlegend,
        )

    # -------------------------------------------------------------------------------------------------------------------
    def scatter(self, x, y, text=None, name="", color=0, opacity=0.4, size=10):
        return go.Scatter(
            name=name,
            x=x,
            y=y,
            text=text,
            mode="markers",
            marker=dict(
                color=self.color(color, opacity + 0.3),  #'rgba(255, 100, 100, 0.9)',
                #                 line=dict(color=self.color(color, opacity), width=1), # 'rgba(150, 50, 50, 0.9)',
                symbol="circle",
                size=size,
                opacity=opacity,
            ),
        )

    # -------------------------------------------------------------------------------------------------------------------
    def scatter3d(self, x, y, z, name="", color=0, opacity=0.8):
        return go.Scatter3d(
            name=name,
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                color=self.color(color, opacity + 0.3),  #'rgba(255, 100, 100, 0.9)',
                #                 line=dict(color=self.color(color, opacity), width=1), # 'rgba(150, 50, 50, 0.9)',
                symbol="circle",
                size=10,
                opacity=opacity,
            ),
        )

    # -------------------------------------------------------------------------------------------------------------------
    def box(self, x, y, name="", color=0, opacity=0.8):
        return go.Box(
            x=x,
            y=y,
            name=name,
            marker=dict(
                color=self.color(color, opacity),  # 'rgba(255, 50, 50, 0.9)', # 'rgb(158,202,225)'
                line=dict(color=self.color(color, opacity), width=1),  # 'rgb(8,48,107)'
            ),
            opacity=opacity,
        )

    # -------------------------------------------------------------------------------------------------------------------
    def color(self, c, opacity=1.0):
        cl.scales["custom"] = [
            "rgba(250, 100, 100, {})".format(opacity),
            "rgba(150, 100, 150, {})".format(opacity),
            "rgba(150, 50, 50, {})".format(opacity),
        ] + ["rgba" + _[3:-1] + ",{})".format(opacity) for _ in cl.scales["11"]["div"]["RdGy"]]
        n_colors = len(cl.scales["custom"])
        return c if type(c) == str else cl.scales["custom"][np.mod(c, n_colors)]


class Video:
    """ Class to make animations
    """

    def __init__(self,framerate=10, xlabel='', x_lim=1, y_lim=1, n_versions=4, colormap=[], txt_pos=0.4):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='')
        self.writer = FFMpegWriter(fps=framerate, metadata=metadata)

        self.fig = plt.figure(figsize=(12,6), dpi=100, facecolor='#202020', edgecolor='#202020')
        plt.margins(10)

        ax = plt.gca()
        self.fig.patch.set_facecolor('#202020')
        ax.set_facecolor('#202020')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('#')
        ax.spines['bottom'].set_color('#AAAAAA')
        ax.spines['top'].set_color('#AAAAAA')
        ax.spines['left'].set_color('#AAAAAA')
        ax.spines['right'].set_color('#AAAAAA')
        ax.xaxis.label.set_color('#AAAAAA')
        ax.tick_params(axis='x', colors='#AAAAAA')

        if colormap == []:
            colorsteps = math.floor(255/n_versions)
            colormap = [rgb2hex(np.array([255,colorsteps*i,colorsteps*i])/255) for i in range(n_versions)]
        self.plts = [plt.plot([], [], colormap[i], linewidth=3)[0] for i in range(n_versions)]
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])
        self.txt_time = plt.figtext(0.2, txt_pos, "", fontsize=12, color='#AAAAAA')
        ax.get_yaxis().set_visible(False)
