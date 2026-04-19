import os
import cv2
import playoffs
import numpy as np
import pandas as pd
import constants as cons
import nhl_utils as nhlu
import math
import datetime as dt

# from predict import predict_season

scale = 0.75
SIZE = int(1200 * scale)
center = (SIZE // 2, SIZE // 2)
# Reused render surface for the radial playoff probability chart.
canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
canvas[:] = (16, 16, 18)


def overlay_center_image(img, image_path, center, size):
    """Overlay an image at the chart center, preserving aspect ratio and alpha.

    Args:
        img: Destination BGR canvas.
        image_path: Path to source image (supports PNG alpha).
        center: (x, y) center point on the destination canvas.
        size: Max width/height box to fit the source image into.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return

    h, w = image.shape[:2]
    scale = min(size / w, size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = center[0] - new_w // 2
    y = center[1] - new_h // 2

    if image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+new_h, x:x+new_w, c] = (
                alpha * image[:, :, c] +
                (1 - alpha) * img[y:y+new_h, x:x+new_w, c]
            )
    else:
        img[y:y+new_h, x:x+new_w] = image


def draw_wedge(img, center, r_outer, r_inner,
               start_angle, end_angle, color):
    """Draw a donut-sector wedge for one team in one playoff round.

    The wedge is formed by sampling points along the outer arc and the inner arc,
    then filling the closed polygon between those arcs.
    """
    step = 1.5  # smoothness

    if not np.isfinite(start_angle) or not np.isfinite(end_angle):
        return

    span = float(end_angle - start_angle)
    if span <= 0:
        return

    # Build both arcs with enough samples so fillPoly always gets a valid contour.
    n_samples = max(2, int(math.ceil(span / step)) + 1)
    outer_angles = np.linspace(start_angle, end_angle, n_samples)
    inner_angles = np.linspace(end_angle, start_angle, n_samples)

    outer_pts = np.column_stack([
        center[0] + r_outer * np.cos(np.radians(outer_angles)),
        center[1] + r_outer * np.sin(np.radians(outer_angles)),
    ])
    inner_pts = np.column_stack([
        center[0] + r_inner * np.cos(np.radians(inner_angles)),
        center[1] + r_inner * np.sin(np.radians(inner_angles)),
    ])

    contour = np.vstack([outer_pts, inner_pts]).astype(np.int32)
    if contour.shape[0] < 3:
        return

    cv2.fillPoly(img, [contour.reshape((-1, 1, 2))], color)


def draw_probability_text(img, center, r_inner, r_outer,
                          start_angle, end_angle, prob):
    """Render a percentage label at the midpoint of a wedge."""

    mid_angle = (start_angle + end_angle) / 2
    rad = math.radians(mid_angle)

    # Place text near the middle of the ring thickness for readability.
    r_text = r_inner + 0.55 * (r_outer - r_inner)

    x = int(center[0] + r_text * math.cos(rad))
    y = int(center[1] + r_text * math.sin(rad))

    text = f"{int(prob*100)}%"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

    cv2.putText(img, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_logo(img, logo_path, x, y, box_size=60):
    """Overlay and center-fit a team logo into a square box on the canvas."""
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return

    h, w = logo.shape[:2]
    scale = min(box_size / w, box_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_offset = x + (box_size - new_w) // 2
    y_offset = y + (box_size - new_h) // 2

    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                alpha * logo[:, :, c] +
                (1 - alpha) * img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )
    else:
        img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = logo


def normalize(probs):
    # Convert absolute probabilities to relative shares so each ring fills 360 degrees.
    probs = np.array(probs)
    return probs / probs.sum()


def compute_angles(probs):
    # Convert normalized probabilities into clockwise angular spans.
    probs = np.array(probs)
    angles = probs * 360

    starts = np.cumsum(np.insert(angles[:-1], 0, 0))
    ends = starts + angles

    return starts, ends


def display_playoff_probability(pred_date, season, playoff_rd=0, matchups=None, display_image=True):
    """Generate and optionally display a radial playoff-probability visualization.

    Workflow:
    1) Load the latest simulation output for the provided date.
    2) Optionally filter/sort teams for a specific playoff round.
    3) Build 5 concentric rings (playoffs -> cup winner).
    4) Draw team wedges, labels, logos, and final center art.
    """

    # list files in output/season_predictions/{pred_date}/ and find the one with the highest n
    season_sched_list = [file for file in os.listdir(cons.season_pred_folder.format(date=pred_date)) if file.startswith(f'season_results_probabilities_{pred_date}_n')]
    if not season_sched_list:
        print(f'No playoff probability files found for {pred_date}. Please run the playoff prediction first.')
        return

    # Multiple simulation files may exist for a date (nX runs).
    # Always use the largest n so visualization reflects the most complete run.
    n_values = [int(file.split('_n')[-1].split('.csv')[0]) for file in season_sched_list]
    n_sims = max(n_values)
    df_prob = pd.read_csv(f'output/season_predictions/{pred_date}/season_results_probabilities_{pred_date}_n{n_sims}.csv')

    # Regular-season view keeps all teams alphabetized.
    if playoff_rd == 0:
        df_prob.sort_values('teamName', inplace=True)
    # Playoff view filters to teams still alive in the selected round,
    # then applies bracket order so neighbors in the wheel match matchups.
    else:
        if playoff_rd == 1:
            df_prob = df_prob.loc[df_prob['playoff_%'] > 0]
        elif playoff_rd == 2:
            df_prob = df_prob.loc[df_prob['make_round_2_%'] > 0]
        elif playoff_rd == 3:
            df_prob = df_prob.loc[df_prob['make_round_3_%'] > 0]
        elif playoff_rd == 4:
            df_prob = df_prob.loc[df_prob['make_cup_final_%'] > 0]
        sort_order = []
        for _, matchup in matchups[playoff_rd].items():
            sort_order.append(matchup.get_team1())
            sort_order.append(matchup.get_team2())
        df_prob['teamName'] = pd.Categorical(df_prob['teamName'], categories=sort_order, ordered=True)
        df_prob.sort_values('teamName', inplace=True)

    # Freeze row order into an ordered categorical so all downstream arrays,
    # colors, wedges, and logos stay aligned by the same team index.
    sort_order = df_prob[cons.team_name_col].tolist()
    df_prob['teamName'] = pd.Categorical(df_prob['teamName'], categories=sort_order, ordered=True)
    df_prob.sort_values('teamName', inplace=True)

    teams = df_prob[cons.team_name_col].tolist()
    logo_paths = {team: cons.team_info[team]['logo'] for team in teams}

    rounds = [
        np.array(list(df_prob['playoff_%']/100)),  # Make playoffs
        np.array(list(df_prob['make_round_2_%']/100)),  # Round 2
        np.array(list(df_prob['make_round_3_%']/100)),  # Round 3
        np.array(list(df_prob['make_cup_final_%']/100)), # Conference finals
        np.array(list(df_prob['win_cup_%']/100)), # Cup win
    ]

    # Store both normalized slices (for geometry) and raw values (for labels).
    rounds = [[normalize(r), r] for r in rounds]

    colors = [cons.team_info[team]['c1'] for team in teams]

    num_teams = len(teams)

    radii = [
        int(450 * scale),
        int(370 * scale),
        int(290 * scale),
        int(210 * scale),
        int(130 * scale),
    ]

    # Constant ring thickness keeps spacing visually uniform across rounds.
    ring_width = int(70 * scale)

    for r_idx, probs in enumerate(rounds):

        r_outer = radii[r_idx]
        r_inner = r_outer - ring_width

        # Geometry uses normalized probabilities so the full ring is always 360.
        starts, ends = compute_angles(probs[0])

        for i in range(num_teams):

            start = starts[i]
            end = ends[i]
            # Labels use raw values (pre-normalization), preserving true percent text.
            prob = probs[1][i]

            draw_wedge(canvas, center, r_outer, r_inner, start, end, colors[i])

            # Skip zero-probability labels to avoid visual noise.
            if prob > 0:
                draw_probability_text(canvas, center, r_inner, r_outer, start, end, prob)

    r_outer = radii[0] + ring_width   # outer boundary of first ring
    logo_radius = r_outer + int(30 * scale)  # slightly outside chart
    logo_size = int(60 * scale)

    # Use first-round shares to anchor logo positions around the outer rim.
    starts, ends = compute_angles(rounds[0][0])

    for i, team in enumerate(teams):

        mid_angle = (starts[i] + ends[i]) / 2
        rad = math.radians(mid_angle)

        x = int(center[0] + logo_radius * math.cos(rad))
        y = int(center[1] + logo_radius * math.sin(rad))

        overlay_logo(canvas, logo_paths[team], x - logo_size // 2, y - logo_size // 2, box_size=logo_size)

    labels = [
        "Make Playoffs",
        "Make Round 2",
        "Make Round 3",
        "Make Cup Final",
        "Win Cup"
    ]

    # Round labels are placed near each ring's top edge for quick legend lookup.
    for i, text in enumerate(labels):
        y = center[1] - radii[i] + 40
        cv2.putText(canvas, text,
                    (center[0] - int(len(text) * 8 * scale), y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # Add cup image in the center and persist the final render to disk.
    overlay_center_image(canvas,'images/stanley_cup.png',center,size=int(120 * scale))

    cv2.imwrite(f'{cons.season_pred_folder.format(date=pred_date)}{cons.playoff_probability_filename.format(season=season, date=pred_date, n=n_sims)}', canvas)
    if display_image:
        cv2.imshow(f"NHL Playoff Probability Wheel for {n_sims} Simulations", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     today_dt = '2026-04-17'
#     season_results_df = predict_season(False, False, today_dt)
#     season_results_points = nhlu.assign_game_points(season_results_df)
#     final_standings_df = nhlu.generate_final_standings(season_results_points, today_dt)
#     _, playoff_matchups = playoffs.playoff_tree_predictions(season_results_df, final_standings_df, False, today_dt, to_csv=False)
#     display_playoff_probability('2026-04-17', '20252026', playoff_rd=1, matchups=playoff_matchups)