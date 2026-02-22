"""
Annotator factory.

Creates and returns all supervision annotator instances used for
drawing detections, labels, and key points on video frames.
"""

import supervision as sv  # type: ignore


def get_annotators():
    """
    Returns a dict of pre-configured supervision annotators.

    Keys:
        ellipse, offside_ellipse, triangle, player_triangle,
        label, offside_label, vertex, vertex_2
    """
    # ── Ellipse annotators (player / offside) ───────────────────────────
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )

    offside_ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#34eb49', '#64f5f0']),
        thickness=2
    )

    # ── Triangle annotators (ball / possession indicator) ───────────────
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )

    player_triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#072D69'),
        base=20, height=17
    )

    # ── Label annotators ────────────────────────────────────────────────
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_scale=0.2,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.BOTTOM_CENTER
    )

    offside_label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#34eb49', '#64f5f0']),
        text_color=sv.Color.from_hex('#000000'),
        text_scale=0.2,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.BOTTOM_CENTER
    )

    # ── Vertex annotators (key-point markers) ───────────────────────────
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8
    )

    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        radius=8
    )

    return {
        "ellipse": ellipse_annotator,
        "offside_ellipse": offside_ellipse_annotator,
        "triangle": triangle_annotator,
        "player_triangle": player_triangle_annotator,
        "label": label_annotator,
        "offside_label": offside_label_annotator,
        "vertex": vertex_annotator,
        "vertex_2": vertex_annotator_2,
    }
