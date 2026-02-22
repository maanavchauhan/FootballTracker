"""
Football AI — main processing pipeline.

Orchestrates the end-to-end workflow:
    1. Load models (detection, field, embeddings)
    2. Fit the team classifier on player crops
    3. Build a per-frame detection record
    4. Clean / interpolate ball positions
    5. Compute player distances and speeds
    6. Render annotated output videos (offside, voronoi, heatmap, pitch)

Usage:
    cd src/
    python main.py
"""

import time
import cv2
import numpy as np
import supervision as sv  # type: ignore

from tqdm import tqdm

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_pitch_voronoi_diagram,
)

# ── Project modules ─────────────────────────────────────────────────────────
from config import SOURCE_VIDEO_PATH, BALL_ID, FPS
from models import load_detection_model, load_field_detection_model

from entity.frame_detections import FrameDetections
from entity.pitch_detections import PitchDetections

from classification.team_classifier import fit_team_classifier

from analysis.ball_tracking import (
    get_ball_positions,
    replace_outliers_based_on_distance,
    interpolate_ball_positions,
)
from analysis.distance_tracking import compute_distances_and_speeds
from analysis.offside import find_last_defender

from annotation.frame_annotator import frame_annotation
from annotation.pitch_annotator import (
    homography_pitch,
    compute_voronoi,
    draw_pitch_heatmap_on_frame,
)

CONFIG = SoccerPitchConfiguration()


# ─── Phase 1: Model & classifier setup ──────────────────────────────────────

def setup():
    """
    Loads all models and fits the team classifier.

    Returns:
        detection_model, field_detection_model, team_classifier
    """
    print("Loading detection model …")
    detection_model = load_detection_model()

    print("Loading field detection model …")
    field_detection_model = load_field_detection_model()

    print("Fitting team classifier …")
    team_classifier = fit_team_classifier(SOURCE_VIDEO_PATH, detection_model)

    return detection_model, field_detection_model, team_classifier


# ─── Phase 2: Build frame record ────────────────────────────────────────────

def build_record(detection_model, field_detection_model, team_classifier):
    """
    Iterates over every frame of the source video, runs detection and
    field models, and stores the results in a list of dicts.
    """
    tracker = sv.ByteTrack()
    tracker.reset()

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    record = []
    frame_number = 0

    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Building record"):
        result = detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        result = field_detection_model.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        f = FrameDetections(frame=frame, detections=detections, tracker=tracker, team_classifier=team_classifier)
        p = PitchDetections(frame, f, key_points=key_points)

        record.append({
            'frame': frame_number,
            'frameDetections': f,
            'pitchDetections': p,
        })
        frame_number += 1

    return record, video_info


# ─── Phase 3: Post-process ball positions ────────────────────────────────────

def postprocess_ball(record):
    """
    Cleans up ball detections: selects the best candidate, removes
    outliers, interpolates gaps, and writes back into the record.
    """
    ball_positions = get_ball_positions(record)
    cleaned_positions = replace_outliers_based_on_distance(ball_positions, 500)
    interpolated_positions = interpolate_ball_positions(cleaned_positions)

    # Saving those positions in the record.
    for frame_info, ball_position in zip(record, interpolated_positions):
        detections = frame_info['frameDetections'].detections
        ball_detections = detections[detections.class_id == BALL_ID]
        other_detections = detections[detections.class_id != BALL_ID]

        if len(ball_detections) == 0 or len(ball_detections) > 1:
            detections = sv.Detections(
                xyxy=np.vstack([other_detections.xyxy, ball_position]),
                confidence=np.hstack([other_detections.confidence, [0.7]]),
                class_id=np.hstack([other_detections.class_id, [BALL_ID]]),
                data={'class_name': np.hstack([other_detections.data['class_name'], ['Ball']])}
            )
            frame_info['frameDetections'].detections = detections
            frame_info['frameDetections'].ball_detections = detections[detections.class_id == BALL_ID]
            frame_info['frameDetections'].ball_detections.xyxy = sv.pad_boxes(
                xyxy=frame_info['frameDetections'].ball_detections.xyxy, px=10
            )


# ─── Phase 4: Render output videos ──────────────────────────────────────────

def render_videos(record, video_info):
    """
    Produces the final annotated video outputs:
        - offside.mp4          (annotated camera view with offside lines)
        - vornoi.mp4           (camera view with voronoi overlay)
        - offside_pitch.mp4    (top-down pitch with offside lines)
        - heatmap.mp4          (standard pitch voronoi)
        - heatmap2.mp4         (custom voronoi diagram)
    """
    pitch_frame = draw_pitch(CONFIG)
    frame_height, frame_width = pitch_frame.shape[:2]

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    video_sink1 = sv.VideoSink("/kaggle/working/offside.mp4", video_info=video_info)
    video_sink2 = sv.VideoSink("/kaggle/working/vornoi.mp4", video_info=video_info)

    video_writer1 = cv2.VideoWriter(
        '/kaggle/working/offside_pitch.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height)
    )
    video_writer2 = cv2.VideoWriter(
        '/kaggle/working/heatmap.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height)
    )
    video_writer3 = cv2.VideoWriter(
        '/kaggle/working/heatmap2.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height)
    )

    previous_possesion = None
    previous_possesion_frame_number = 0
    record2 = []

    # Statistics
    team0possesion = 0
    team1possesion = 0
    team0passes = 0
    team1passes = 0

    with video_sink1, video_sink2:
        for idx, (frame, frame_info) in enumerate(
            tqdm(zip(frame_generator, record), total=video_info.total_frames, desc="Rendering")
        ):
            f = frame_info['frameDetections']
            p = frame_info['pitchDetections']
            frame_number = frame_info['frame']

            last_defender_positions, potential_offsides = find_last_defender(f, p)

            output, possesion = frame_annotation(
                frame, f,
                offside=True,
                last_defender_positions=last_defender_positions,
                potential_offsides=potential_offsides,
                previous_possesion=previous_possesion,
                cumulative_distances=frame_info['cumulative_distances'],
                instantaneous_speed=frame_info['instantaneous_speed'],
                p=p,
            )

            # ── Possession / offside bookkeeping ────────────────────────
            if possesion is not None:
                if possesion.tracker_id in potential_offsides:
                    prev_frame_info = record2[(frame_number - previous_possesion_frame_number)]
                    if (
                        possesion.tracker_id in prev_frame_info['potential_offsides']
                        and prev_frame_info['player'].class_id == possesion.tracker_id
                    ):
                        print('OFFSIDE! Frame Number: ' + str(prev_frame_info['frame']))

                if possesion.class_id == 0:
                    team0possesion += 1
                    if previous_possesion is not None and previous_possesion.class_id == 0:
                        team0passes += 1
                else:
                    team1possesion += 1
                    if previous_possesion is not None and previous_possesion.class_id == 1:
                        team1passes += 1

                previous_possesion = possesion
                previous_possesion_frame_number = frame_number

            record2.append({
                'frame': frame_number,
                'player': possesion,
                'potential_offsides': potential_offsides,
            })

            # ── Voronoi overlays ────────────────────────────────────────
            voronoi_diagram = compute_voronoi(f, p)
            voronoi_frame = draw_pitch_heatmap_on_frame(voronoi_diagram, p.transformer_inverse, frame)

            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=p.pitch_players_xy[f.players_detections.class_id == 0],
                team_2_xy=p.pitch_players_xy[f.players_detections.class_id == 1],
                team_1_color=sv.Color.from_hex('00BFFF'),
                team_2_color=sv.Color.from_hex('FF1493'),
                pitch=annotated_frame
            )

            # ── Write frames ────────────────────────────────────────────
            video_sink1.write_frame(output)
            video_sink2.write_frame(voronoi_frame)
            video_writer1.write(
                homography_pitch(frame, f, p,
                                 offside=True,
                                 last_defender_positions=last_defender_positions,
                                 potential_offsides=potential_offsides)
            )
            video_writer2.write(annotated_frame)
            video_writer3.write(voronoi_diagram)

    video_writer1.release()
    video_writer2.release()
    video_writer3.release()

    return team0possesion, team1possesion, team0passes, team1passes


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # 1. Load models and fit classifier
    detection_model, field_detection_model, team_classifier = setup()

    # 2. Build frame-by-frame detection record
    record, video_info = build_record(detection_model, field_detection_model, team_classifier)

    # 3. Post-process ball positions (clean, interpolate)
    postprocess_ball(record)

    # 4. Compute player distances and speeds
    compute_distances_and_speeds(record)

    # 5. Render all output videos
    t0poss, t1poss, t0pass, t1pass = render_videos(record, video_info)

    elapsed = time.time() - start_time
    print(f"--- {elapsed:.1f} seconds ---")
    print("---PASSES:---")
    print(f"Team 0: {t0pass}")
    print(f"Team 1: {t1pass}")
    print("---POSSESSION:---")
    total = t0poss + t1poss
    if total > 0:
        print(f"Team 0: {t0poss * 100 / total:.1f}%")
        print(f"Team 1: {t1poss * 100 / total:.1f}%")


if __name__ == "__main__":
    main()
