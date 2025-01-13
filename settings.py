COLORDICT = {
    "HC": "tab:green",
    "PD": "tab:red",
    "AD": "tab:purple",
    "DLB": "cornflowerblue",
    "A": "tab:green",
    "B": "yellowgreen",
    "C": "darkgreen",
}

MARKERDICT = {
    "HC": "D",
    "PD": "^",
    "AD": "o",
    "DLB": "D",
    "A": "^",
    "B": "^",
    "C": ">",
}

DEFAULT_MARKER_STYLE = {
    "color": "tab:green",
    "marker": "D",
    "edgecolors": "white",
    "s": 50,
    "alpha": 0.7,
    "linewidths": 0.2,
}

def _update_style_dict(default_marker_style: dict, add_marker_style: dict) -> dict:
    marker_style = default_marker_style.copy()
    marker_style.update(add_marker_style)
    return marker_style


MARKER_STYLES = {
   
    "HC": _update_style_dict(
        DEFAULT_MARKER_STYLE, {"color": COLORDICT["HC"], "marker": MARKERDICT["HC"]}
    ),
    "PD": _update_style_dict(
        DEFAULT_MARKER_STYLE, {"color": COLORDICT["PD"], "marker": MARKERDICT["PD"]}
    ),
    "AD": _update_style_dict(
        DEFAULT_MARKER_STYLE, {"color": COLORDICT["AD"], "marker": MARKERDICT["AD"]}
    ),
    "DLB": _update_style_dict(
        DEFAULT_MARKER_STYLE, {"color": COLORDICT["DLB"], "marker": MARKERDICT["DLB"]}
    ),
    "A": _update_style_dict(
        DEFAULT_MARKER_STYLE,
        {"color": COLORDICT["A"], "marker": MARKERDICT["A"]},
    ),
    "B": _update_style_dict(
        DEFAULT_MARKER_STYLE,
        {"color": COLORDICT["B"], "marker": MARKERDICT["B"]},
    ),
   "C": _update_style_dict(
        DEFAULT_MARKER_STYLE,
        {"color": COLORDICT["C"], "marker": MARKERDICT["C"]},
    ),
}

DEFAULT_PROTOTYPE_MARKER_STYLE = {
    "color": "tab:green",
    "marker": "D",
    "edgecolors": "black",
    "s": 100,
    "alpha": 0.9,
    "linewidths": 2.0,
}


PROTOTYPE_MARKER_STYLE = {
    "HC": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE, {"color": COLORDICT["HC"], "marker": MARKERDICT["HC"]}
    ),
    "PD": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE, {"color": COLORDICT["PD"], "marker": MARKERDICT["PD"]}
    ),
    "AD": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE, {"color": COLORDICT["AD"], "marker": MARKERDICT["AD"]}
    ),
    "DLB": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE, {"color": COLORDICT["DLB"], "marker": MARKERDICT["DLB"]}
    ),
    "A": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE,
        {"color": COLORDICT["A"], "marker": MARKERDICT["A"]},
    ),
    "B": _update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE,
        {"color": COLORDICT["B"], "marker": MARKERDICT["B"]},
    ),
    "C":_update_style_dict(
        DEFAULT_PROTOTYPE_MARKER_STYLE,
        {"color": COLORDICT["C"], "marker": MARKERDICT["C"]},
    ),
}

DEFAULT_LEGEND_STYLE = {
    "mfc": "tab:green",
    "marker": "D",
    "color": "w",
    "markersize": 6,
    "markeredgewidth": 0,
}

LEGEND_MARKERS = {
    
    "HC": _update_style_dict(
        DEFAULT_LEGEND_STYLE, {"mfc": COLORDICT["HC"], "marker": MARKERDICT["HC"]}
    ),
    "PD": _update_style_dict(
        DEFAULT_LEGEND_STYLE, {"mfc": COLORDICT["PD"], "marker": MARKERDICT["PD"]}
    ),
    "AD": _update_style_dict(
        DEFAULT_LEGEND_STYLE, {"mfc": COLORDICT["AD"], "marker": MARKERDICT["AD"]}
    ),
    "DLB": _update_style_dict(
        DEFAULT_LEGEND_STYLE, {"mfc": COLORDICT["DLB"], "marker": MARKERDICT["DLB"]}
    ),
    "A": _update_style_dict(
        DEFAULT_LEGEND_STYLE,
        {"mfc": COLORDICT["A"], "marker": MARKERDICT["A"]},
    ),  
    "B": _update_style_dict(
        DEFAULT_LEGEND_STYLE,
        {"mfc": COLORDICT["B"], "marker": MARKERDICT["B"]},
    ),
    "C": _update_style_dict(
        DEFAULT_LEGEND_STYLE,
        {"mfc": COLORDICT["C"], "marker": MARKERDICT["C"]},
    ),
}
