import matplotlib.pyplot as plt
import ipywidgets as widgets
import pandas as pd
import numpy as np
from ipywidgets import interact, fixed
from typing import List, Literal, Optional, Union

def _update_plot(
    data: pd.DataFrame, 
    seg_num: int, 
    x: str, 
    y: Union[str, List[str]], 
    beats: Optional[str] = None,
    artifacts: Optional[str] = None,
    title: Optional[str] = None, 
    color: Optional[Union[str, List[str]]] = None,
) -> None:
    """Update the plot of a data segment."""
    
    plt.rcParams['font.size'] = 10
    plt.figure(figsize = (14, 2.5))

    if color is None:
        colors = plt.cm.Set2.colors
    elif isinstance(color, str):
        colors = [color] * (len(y) if isinstance(y, (list, tuple)) else 1)
    elif isinstance(color, list):
        colors = color

    # Select segment data
    segment = data.loc[data.Segment == seg_num]

    # Handle single or multiple y-values
    if isinstance(y, str):
        plt.plot(segment[x], segment[y], label = y, lw = 1.3, color = colors[0], zorder = 2)
    elif isinstance(y, (list, tuple)):
        for yval, c in zip(y, colors):
            plt.plot(segment[x], segment[yval], label = yval, lw = 1.3, color = c, zorder = 2)

    # Scatter plot for beats
    if beats is not None:
        plt.scatter(
            segment.loc[segment[beats] == 1, x], 
            segment.loc[segment[beats] == 1, y[0] if isinstance(y, (list, tuple)) else y], 
            s = 15, color = 'royalblue', linewidth = 0.5, label = 'Beat', zorder = 3
        )
    if artifacts is not None:
        plt.scatter(
            segment.loc[segment[artifacts] == 1, x], 
            segment.loc[segment[artifacts] == 1, y[0] if isinstance(y, (list, tuple)) else y], 
            s = 15, color = 'orange', linewidth = 0.5, label = 'Artifact', zorder = 3
        )

    # Figure parameters
    plt.xlim(segment[x].iloc[0], segment[x].iloc[-1])
    plt.title(title if title else f'Segment {seg_num}')
    plt.grid(axis = 'both', linestyle = ':')

    # Show legend if necessary
    if beats or isinstance(y, (list, tuple)):
        plt.legend(frameon = False, loc = 'upper left', bbox_to_anchor = (1, 0.7), fontsize = 9)

    # Remove all borders
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout() 
    plt.show()

def interactive_plot(
    data: pd.DataFrame, 
    x: str, 
    y: Union[str, List[str]], 
    beats: Optional[str] = None,
    artifacts: Optional[str] = None,
    title: Optional[str] = None, 
    color: Optional[Union[str, List[str]]] = None,
) -> plt.figure:
    """Create a plot with an interactive slider."""
    return interact(
        _update_plot, 
        data = fixed(data), 
        x = fixed(x), 
        y = fixed(y), 
        beats = fixed(beats), 
        artifacts = fixed(artifacts),
        title = fixed(title),
        color = fixed(color),
        seg_num = widgets.IntSlider(
            min = 1, max = max(data.Segment), step = 1, value = 1, 
            description = 'Segment:')
    );

def summary_plot(
    data: pd.DataFrame, 
    x: str, 
    y: Union[str, List[str]],
    kind: Literal['bar', 'line'] = 'bar',
    title: Optional[str] = None,
    legend: bool = True,
) -> None:
    """Create a plot summarizing the data quality."""
    
    plt.rcParams['font.size'] = 10
    plt.figure(figsize = (8, 3))
    colors = plt.cm.Set2.colors

    # If multiple y values are passed
    if isinstance(y, list):
        if kind == 'bar':
            bar_width = 0.35
            x_pos = np.arange(len(data[x])) 
            for i, col in enumerate(y):
                plt.bar(
                    x_pos + i * bar_width, 
                    data[col], 
                    color = colors[i+1],
                    width = bar_width, 
                    label = col, 
                    zorder = 3)
            plt.xticks(x_pos + bar_width / 2, data[x])
        elif kind == 'line':
            for i, col in enumerate(y):
                plt.plot(
                    data[x], 
                    data[col], 
                    color = colors[i+1], 
                    lw = 1.2, 
                    label = col, 
                    zorder = 3)
    else:
        if kind == 'bar':
            plt.bar(
                data[x], data[y], 
                color = colors[0], 
                width = 0.6, 
                zorder = 2, 
                label = y)
        elif kind == 'line':
            plt.plot(
                data[x], data[y], 
                color = colors[0], 
                lw = 1.2, 
                zorder = 2, 
                label = y)
        plt.xticks(np.arange(data[x].iloc[0], data[x].iloc[-1] + 1))

    # Show legend 
    if legend:
        plt.legend(frameon = False, fontsize = 9, loc = 'best')
    ax = plt.gca()

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(x)
    plt.grid(axis = 'y', ls = ':')
    if title is not None:
        plt.title(title)
    plt.tight_layout()