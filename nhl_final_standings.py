
import cv2
import numpy as np
import pandas as pd
import constants as cons
import os


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r"output\\season_predictions\\{asofdate}\\regularseason_standings_{asofdate}.csv"
LOGO_DIR = "images"
OUTPUT_PATH = "output\\season_predictions\\{asofdate}\\regularseason_standings_{asofdate}.png"

# Overall graphic scale
SCALE = 0.85

# Base canvas width before SCALE is applied
BASE_WIDTH = 2200

# ------------------------------------------------------------
# Layout tuning
# ------------------------------------------------------------

OUTER_MARGIN = 70
CONFERENCE_GAP = 55

CONFERENCE_TITLE_HEIGHT = 72

DIVISION_HEADER_HEIGHT = 58
COLUMN_HEADER_HEIGHT = 38
TEAM_ROW_HEIGHT = 82

DIVISION_GAP = 22

LEGEND_HEIGHT = 70
BOTTOM_MARGIN = 22

# Width reserved for GP/W/L/OTL/PTS/P%
STATS_AREA_WIDTH = 485

# Minimum space between team name and GP
TEAM_STATS_GAP = 28

# Logo box between rank and team name
LOGO_BOX_SIZE = 58


# ============================================================
# COLORS - BGR
# ============================================================

BACKGROUND = (9, 18, 30)
PANEL = (13, 28, 43)
PANEL_ALT = (15, 33, 50)
PANEL_HEADER = (20, 55, 82)

WHITE = (245, 247, 250)
LIGHT_GRAY = (190, 201, 214)
GRAY = (115, 130, 145)

BLUE = (25, 145, 225)
BLUE_BRIGHT = (40, 175, 245)
BLUE_DARK = (12, 74, 115)

CONFERENCE_LEADER = (30, 175, 255)
DIVISIONAL_SPOT_COLOR = (0, 255, 0)
WILDCARD_SPOT_COLOR = (255, 0, 0)
ELIMINATED_COLOR = (175, 185, 198)

GRID = (27, 57, 79)
BORDER = (24, 116, 175)

TITLE_ACCENT = (35, 145, 225)


# ============================================================
# TEAM LOGOS
# ============================================================

TEAM_LOGOS = {key: value['logo'] for key, value in cons.team_info.items()}


# ============================================================
# SCALING
# ============================================================

def s(value):
    return int(round(value * SCALE))


# ============================================================
# TEXT
# ============================================================

def get_font(size, thickness=1):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(size * SCALE)
    font_thickness = max(1, int(round(thickness * SCALE)))
    return font_face, font_scale, font_thickness


def put_text(
    img,
    text,
    position,
    font_size,
    color,
    thickness=1,
    align="left"
):
    """
    Safe wrapper around cv2.putText.
    """

    text = str(text)

    font_face, font_scale, font_thickness = get_font(
        font_size,
        thickness
    )

    x = int(round(position[0]))
    y = int(round(position[1]))

    if align == "center":
        text_width = cv2.getTextSize(
            text,
            font_face,
            float(font_scale),
            int(font_thickness)
        )[0][0]

        x -= text_width // 2

    cv2.putText(
        img,
        text,
        (x, y),
        font_face,
        float(font_scale),
        color,
        int(font_thickness),
        cv2.LINE_AA
    )


def text_width(text, font_size, thickness=1):
    font_face, font_scale, font_thickness = get_font(
        font_size,
        thickness
    )

    return cv2.getTextSize(
        str(text),
        font_face,
        float(font_scale),
        int(font_thickness)
    )[0][0]


# ============================================================
# LOGO
# ============================================================

def overlay_logo(
    img,
    logo_path,
    center_x,
    center_y,
    box_size=LOGO_BOX_SIZE
):
    """
    Draw a logo centered inside a box while preserving aspect
    ratio and PNG transparency.
    """

    if not os.path.exists(logo_path):
        return

    logo = cv2.imread(
        logo_path,
        cv2.IMREAD_UNCHANGED
    )

    if logo is None:
        return

    h, w = logo.shape[:2]

    if h <= 0 or w <= 0:
        return

    box_size = s(box_size)

    resize_scale = min(
        box_size / w,
        box_size / h
    )

    new_w = max(1, int(round(w * resize_scale)))
    new_h = max(1, int(round(h * resize_scale)))

    logo = cv2.resize(
        logo,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    x = int(round(center_x - new_w / 2))
    y = int(round(center_y - new_h / 2))

    if (
        x < 0
        or y < 0
        or x + new_w > img.shape[1]
        or y + new_h > img.shape[0]
    ):
        return

    if len(logo.shape) == 3 and logo.shape[2] == 4:

        alpha = logo[:, :, 3].astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]

        foreground = logo[:, :, :3].astype(float)

        background = img[
            y:y + new_h,
            x:x + new_w
        ].astype(float)

        blended = (
            foreground * alpha
            + background * (1 - alpha)
        )

        img[
            y:y + new_h,
            x:x + new_w
        ] = blended.astype(np.uint8)

    else:

        img[
            y:y + new_h,
            x:x + new_w
        ] = logo[:, :, :3]


# ============================================================
# DECORATIVE SHAPES
# ============================================================

def draw_slanted_accent(
    img,
    x,
    y,
    height,
    width,
    color
):
    """
    Draw the angled broadcast-style accent used beside
    conference/division headings.
    """

    x = int(x)
    y = int(y)
    height = int(height)
    width = int(width)

    pts = np.array([
        [x + width // 2, y],
        [x + width, y],
        [x + width // 2, y + height],
        [x, y + height]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [pts],
        color
    )


def draw_panel_border(
    img,
    x,
    y,
    width,
    height
):
    cv2.rectangle(
        img,
        (x, y),
        (x + width, y + height),
        BORDER,
        max(1, s(1))
    )


# ============================================================
# DATA
# ============================================================

def load_standings(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find standings CSV:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    numeric_columns = [
        "conferenceSeed",
        "divisionSeed",
        "totalGames",
        "totalWins",
        "totalLosses",
        "totalOTLs",
        "totalPoints",
        "pointsPercentage",
        "totalRegWins",
        "totalRegOTWins",
        "goalDifferential",
        "totalGoalsFor",
    ]

    df['playoffSeed'] = pd.to_numeric(df['playoffSeed'].str[-1:],
                errors="coerce")

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def get_division_order(conference):

    if conference == "Eastern":
        return [
            "Atlantic",
            "Metropolitan"
        ]

    return [
        "Central",
        "Pacific"
    ]


# ============================================================
# STATUS
# ============================================================

def get_team_status(row):

    conference_seed = row["conferenceSeed"]
    division_seed = row["divisionSeed"]
    playoff_seed = row["playoffSeed"]

    if (
        pd.notna(conference_seed)
        and int(conference_seed) == 1
    ):
        return "conference_leader"

    if (
        pd.notna(division_seed)
        and int(division_seed) <= 3
    ):
        return "divisional_spot"

    if (
        pd.notna(playoff_seed)
        and int(playoff_seed) <= 2
    ):
        return "wildcard_spot"

    return "outside"


def get_status_color(row):

    status = get_team_status(row)

    if status == "conference_leader":
        return CONFERENCE_LEADER

    if status == "divisional_spot":
        return DIVISIONAL_SPOT_COLOR

    if status == "wildcard_spot":
        return WILDCARD_SPOT_COLOR

    return ELIMINATED_COLOR


# ============================================================
# LAYOUT
# ============================================================

def get_stats_positions(
    x,
    width
):
    """
    Stats are kept together on the far-right side.
    """

    stats_left = (
        x
        + width
        - s(STATS_AREA_WIDTH)
    )

    return {
        "gp": stats_left,
        "w": stats_left + s(78),
        "l": stats_left + s(156),
        "otl": stats_left + s(234),
        "pts": stats_left + s(315),
        "pct": stats_left + s(410),
    }


def get_team_name_area(
    x,
    width
):
    """
    Returns the usable team-name area between the logo and
    the statistics.
    """

    team_x = x + s(220)

    stats = get_stats_positions(
        x,
        width
    )

    max_width = (
        stats["gp"]
        - team_x
        - s(TEAM_STATS_GAP)
    )

    return team_x, max_width


# ============================================================
# HEIGHT
# ============================================================

def calculate_conference_height(df):

    if df.empty:
        return s(CONFERENCE_TITLE_HEIGHT)

    height = s(CONFERENCE_TITLE_HEIGHT)

    conference = df["conferenceName"].iloc[0]

    for division in get_division_order(conference):

        division_df = df[
            df["divisionName"] == division
        ]

        if division_df.empty:
            continue

        height += s(DIVISION_HEADER_HEIGHT)
        height += s(COLUMN_HEADER_HEIGHT)
        height += (
            len(division_df)
            * s(TEAM_ROW_HEIGHT)
        )
        height += s(DIVISION_GAP)

    return height


def calculate_canvas_height(df):

    eastern_height = calculate_conference_height(
        df[df["conferenceName"] == "Eastern"]
    )

    western_height = calculate_conference_height(
        df[df["conferenceName"] == "Western"]
    )

    title_area = s(145)

    conference_area = max(
        eastern_height,
        western_height
    )

    legend_area = s(LEGEND_HEIGHT)

    bottom_margin = s(BOTTOM_MARGIN)

    return (
        title_area
        + conference_area
        + legend_area
        + bottom_margin
    )


# ============================================================
# TEAM NAME
# ============================================================

def draw_team_name(
    img,
    team_name,
    x,
    baseline,
    max_width
):
    """
    Shrinks long team names just enough to keep them inside
    their column.
    """

    # special Montreal clause
    if team_name[:3] == 'Mon':
        team_name = 'Montreal Canadiens'

    font_size = 0.82
    minimum_size = 0.55

    while (
        font_size > minimum_size
        and text_width(
            team_name,
            font_size,
            1
        ) > max_width
    ):
        font_size -= 0.025

    font_size = max(
        minimum_size,
        font_size
    )

    put_text(
        img,
        team_name,
        (x, baseline),
        font_size,
        WHITE,
        thickness=1
    )


# ============================================================
# TEAM ROW
# ============================================================

def draw_team_row(
    img,
    row,
    x,
    y,
    width
):

    row_height = s(TEAM_ROW_HEIGHT)

    # --------------------------------------------------------
    # Row background
    # --------------------------------------------------------

    cv2.rectangle(
        img,
        (x, y),
        (x + width, y + row_height),
        PANEL,
        -1
    )

    # --------------------------------------------------------
    # Status stripe
    # --------------------------------------------------------

    cv2.rectangle(
        img,
        (x, y),
        (x + s(6), y + row_height),
        get_status_color(row),
        -1
    )

    # --------------------------------------------------------
    # Bottom row line
    # --------------------------------------------------------

    cv2.line(
        img,
        (x, y + row_height - s(1)),
        (x + width, y + row_height - s(1)),
        GRID,
        max(1, s(1))
    )

    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    rank_x = x + s(42)
    logo_x = x + s(122)

    team_x, team_max_width = get_team_name_area(
        x,
        width
    )

    stats = get_stats_positions(
        x,
        width
    )

    baseline = y + s(52)

    # --------------------------------------------------------
    # DIVISION RANK
    # --------------------------------------------------------

    division_seed = row["divisionSeed"]

    if pd.notna(division_seed):
        rank = str(int(division_seed))
    else:
        rank = "-"

    put_text(
        img,
        rank,
        (rank_x, baseline),
        0.88,
        WHITE,
        thickness=2,
        align="center"
    )

    # --------------------------------------------------------
    # TEAM LOGO
    # --------------------------------------------------------

    team_name = str(row["teamName"])

    logo_filename = TEAM_LOGOS.get(team_name)

    if logo_filename:

        logo_path = os.path.join(
            logo_filename
        )

        overlay_logo(
            img,
            logo_path,
            logo_x,
            y + row_height // 2,
            LOGO_BOX_SIZE
        )

    # --------------------------------------------------------
    # TEAM NAME
    # --------------------------------------------------------

    draw_team_name(
        img,
        team_name,
        team_x,
        baseline,
        team_max_width
    )

    # --------------------------------------------------------
    # GP / W / L / OTL
    # --------------------------------------------------------

    put_text(
        img,
        int(row["totalGames"]),
        (stats["gp"], baseline),
        0.70,
        LIGHT_GRAY,
        align="center"
    )

    put_text(
        img,
        int(row["totalWins"]),
        (stats["w"], baseline),
        0.70,
        LIGHT_GRAY,
        align="center"
    )

    put_text(
        img,
        int(row["totalLosses"]),
        (stats["l"], baseline),
        0.70,
        LIGHT_GRAY,
        align="center"
    )

    put_text(
        img,
        int(row["totalOTLs"]),
        (stats["otl"], baseline),
        0.70,
        LIGHT_GRAY,
        align="center"
    )

    # --------------------------------------------------------
    # POINTS
    # --------------------------------------------------------

    put_text(
        img,
        int(row["totalPoints"]),
        (stats["pts"], baseline),
        0.78,
        WHITE,
        thickness=2,
        align="center"
    )

    # --------------------------------------------------------
    # POINT PERCENTAGE
    # --------------------------------------------------------

    pct = row["pointsPercentage"]

    if pd.notna(pct):
        pct_text = f"{pct * 100:.1f}%"
    else:
        pct_text = "-"

    put_text(
        img,
        pct_text,
        (stats["pct"], baseline),
        0.66,
        LIGHT_GRAY,
        align="center"
    )


# ============================================================
# COLUMN HEADERS
# ============================================================

def draw_column_headers(
    img,
    x,
    y,
    width
):

    stats = get_stats_positions(
        x,
        width
    )

    baseline = y + s(27)

    headers = [
        ("GP", "gp"),
        ("W", "w"),
        ("L", "l"),
        ("OTL", "otl"),
        ("PTS", "pts"),
        ("P%", "pct"),
    ]

    for label, key in headers:

        put_text(
            img,
            label,
            (stats[key], baseline),
            0.50,
            LIGHT_GRAY,
            thickness=1,
            align="center"
        )


# ============================================================
# DIVISION
# ============================================================

def draw_division_section(
    img,
    division_df,
    division_name,
    x,
    y,
    width
):

    header_h = s(DIVISION_HEADER_HEIGHT)
    column_h = s(COLUMN_HEADER_HEIGHT)
    row_h = s(TEAM_ROW_HEIGHT)

    # --------------------------------------------------------
    # Division header
    # --------------------------------------------------------

    cv2.rectangle(
        img,
        (x+6, y),
        (x + width, y + header_h),
        PANEL_HEADER,
        -1
    )

    draw_panel_border(
        img,
        x+6,
        y,
        width-6,
        header_h
    )

    # Angled accent
    draw_slanted_accent(
        img,
        x,
        y,
        header_h,
        s(18),
        BLUE_BRIGHT
    )

    put_text(
        img,
        f"{division_name.upper()} DIVISION",
        (x + s(36), y + s(39)),
        0.78,
        WHITE,
        thickness=2
    )

    # --------------------------------------------------------
    # Column header strip
    # --------------------------------------------------------

    column_y = y + header_h

    cv2.rectangle(
        img,
        (x, column_y),
        (
            x + width,
            column_y + column_h
        ),
        BACKGROUND,
        -1
    )

    draw_column_headers(
        img,
        x,
        column_y,
        width
    )

    # --------------------------------------------------------
    # Teams
    # --------------------------------------------------------

    current_y = column_y + column_h

    division_df = division_df.sort_values(
        by="divisionSeed",
        ascending=True,
        kind="stable"
    )

    for _, row in division_df.iterrows():

        draw_team_row(
            img,
            row,
            x,
            current_y,
            width
        )

        current_y += row_h

    # --------------------------------------------------------
    # Division bottom gap
    # --------------------------------------------------------

    current_y += s(DIVISION_GAP)

    return current_y


# ============================================================
# CONFERENCE
# ============================================================

def draw_conference(
    img,
    df,
    conference,
    x,
    y,
    width
):

    title_h = s(CONFERENCE_TITLE_HEIGHT)

    # --------------------------------------------------------
    # Conference title
    # --------------------------------------------------------

    # Large angled conference accent
    draw_slanted_accent(
        img,
        x,
        y + s(8),
        s(54),
        s(25),
        TITLE_ACCENT
    )

    put_text(
        img,
        f"{conference.upper()} CONFERENCE",
        (x + s(45), y + s(48)),
        1.05,
        WHITE,
        thickness=2
    )

    current_y = y + title_h

    # --------------------------------------------------------
    # Divisions
    # --------------------------------------------------------

    for division in get_division_order(conference):

        division_df = df[
            df["divisionName"] == division
        ]

        if division_df.empty:
            continue

        current_y = draw_division_section(
            img,
            division_df,
            division,
            x,
            current_y,
            width
        )

    return current_y


# ============================================================
# TITLE
# ============================================================

def draw_title(img, season_name, subtitle):

    center_x = img.shape[1] // 2

    title = f"{season_name} NHL REGULAR SEASON STANDINGS"

    put_text(
        img,
        title,
        (center_x, s(65)),
        1.35,
        WHITE,
        thickness=2,
        align="center"
    )

    subtitle_width = text_width(
        subtitle,
        0.65,
        1
    )

    line_y = s(94)

    # Left line
    cv2.line(
        img,
        (
            center_x - subtitle_width // 2 - s(28),
            line_y
        ),
        (
            center_x - subtitle_width // 2 - s(8),
            line_y
        ),
        BLUE,
        max(1, s(2))
    )

    # Right line
    cv2.line(
        img,
        (
            center_x + subtitle_width // 2 + s(8),
            line_y
        ),
        (
            center_x + subtitle_width // 2 + s(28),
            line_y
        ),
        BLUE,
        max(1, s(2))
    )

    put_text(
        img,
        subtitle,
        (center_x, s(105)),
        0.65,
        LIGHT_GRAY,
        thickness=1,
        align="center"
    )


# ============================================================
# LEGEND
# ============================================================

def draw_legend(
    img,
    x,
    y
):

    items = [
        (CONFERENCE_LEADER, "Conference Leader"),
        (DIVISIONAL_SPOT_COLOR, "Divisional Spot"),
        (WILDCARD_SPOT_COLOR, "Wildcard Spot"),
        (ELIMINATED_COLOR, "Missed Playoffs"),
    ]

    current_x = x

    for color, label in items:

        # Circle instead of square for a more broadcast-style key.
        center = (
            current_x + s(12),
            y + s(11)
        )

        cv2.circle(
            img,
            center,
            s(10),
            color,
            -1
        )

        put_text(
            img,
            label,
            (
                current_x + s(34),
                y + s(17)
            ),
            0.58,
            LIGHT_GRAY,
            thickness=1
        )

        current_x += s(205)


# ============================================================
# MAIN
# ============================================================

def main(asofdate, season_name, reg_season_over):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_standings(
        CSV_PATH.format(asofdate=asofdate)
    )

    # --------------------------------------------------------
    # Canvas
    # --------------------------------------------------------

    width = s(BASE_WIDTH)

    height = calculate_canvas_height(
        df
    )

    img = np.full(
        (height, width, 3),
        BACKGROUND,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if reg_season_over:
        subtitle = "FINAL REGULAR SEASON RESULTS"
    else:
        subtitle = f"PREDICTED REGULAR SEASON RESULTS (as of {asofdate})"

    draw_title(img, season_name,subtitle)

    # --------------------------------------------------------
    # Conference dimensions
    # --------------------------------------------------------

    margin_x = s(OUTER_MARGIN)

    gap = s(CONFERENCE_GAP)

    conference_width = (
        width
        - 2 * margin_x
        - gap
    ) // 2

    eastern_x = margin_x

    western_x = (
        eastern_x
        + conference_width
        + gap
    )

    conference_y = s(132)

    # --------------------------------------------------------
    # Draw conferences
    # --------------------------------------------------------

    eastern_bottom = draw_conference(
        img,
        df[df["conferenceName"] == "Eastern"],
        "Eastern",
        eastern_x,
        conference_y,
        conference_width
    )

    western_bottom = draw_conference(
        img,
        df[df["conferenceName"] == "Western"],
        "Western",
        western_x,
        conference_y,
        conference_width
    )

    # --------------------------------------------------------
    # Center divider
    # --------------------------------------------------------

    separator_x = (
        eastern_x
        + conference_width
        + gap // 2
    )

    divider_top = conference_y + s(12)

    divider_bottom = (
        max(
            eastern_bottom,
            western_bottom
        )
        - s(DIVISION_GAP)
    )

    cv2.line(
        img,
        (separator_x, divider_top),
        (separator_x, divider_bottom),
        BLUE_DARK,
        max(1, s(2))
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_y = (
        max(
            eastern_bottom,
            western_bottom
        )
        + s(1)
    )

    draw_legend(
        img,
        margin_x,
        legend_y
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if not cv2.imwrite(
        OUTPUT_PATH.format(asofdate=asofdate),
        img
    ):
        raise RuntimeError(
            f"Failed to save image: {OUTPUT_PATH.format(asofdate=asofdate)}"
        )

    print("\nSaving final standings graphic... ")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main(
        asofdate="2026-06-14",
        season_name="2025-2026",
        reg_season_over=True
        )