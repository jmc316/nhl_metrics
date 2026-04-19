import cv2

import numpy as np
import constants as cons

# local constants
WIDTH, HEIGHT = 1400, 800
# Shared canvas reused for one bracket render.
CANVAS = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
CANVAS[:] = (12, 12, 14)
# Column anchors for bracket rounds (left conference).
LEFT_X = [20, 270, 410]
# Mirror left anchors around the center line for the right conference.
RIGHT_X = [WIDTH-LEFT_X[0], WIDTH-LEFT_X[1], WIDTH-LEFT_X[2]]
CENTER_X = WIDTH // 2
# First round cards are wider to include seed text; later rounds are compact.
CARD_1_W, CARD_1_H = 150, 65
CARD_W, CARD_H = 100, 65
LOGO_SIZE_CARD = 50
LOGO_SIZE_CHAMP = 150
LOGO_SIZE_CUP = 500
# Vertical layout controls for each round.
BASE_Y = 150
R1_SPACE = 80
R2_SPACE = R1_SPACE * 2
R3_SPACE = R2_SPACE * 2


def draw_glow_line(canvas, pt1, pt2, color=(0,180,255)):
    """Draw a connector line with a soft glow effect behind a bright core."""
    overlay = canvas.copy()
    cv2.line(overlay, pt1, pt2, color, 6)
    canvas[:] = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    cv2.line(canvas, pt1, pt2, (240,240,240), 2)


def draw_series_score(canvas, x, y, text):
    """Draw a small score badge centered at (x, y)."""
    w, h = 70, 30
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x-w//2, y-h//2), (x+w//2, y+h//2), (25,25,25), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    cv2.rectangle(canvas, (x-w//2, y-h//2), (x+w//2, y+h//2), (80,80,80), 1)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
    cv2.putText(canvas, text, (x - tw//2, y + th//2),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (240,240,240), 1, cv2.LINE_AA)


def overlay_logo(canvas, path, x, y, logo_size, pad_color=(0, 0, 0)):
    """Overlay a team logo into a fixed-height card slot.

    The source image is resized with preserved aspect ratio, padded to a square,
    then cropped so the visible area fits the card height convention used here.
    """
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return
    
    h, w = logo.shape[:2]
    target_h, target_w = (logo_size*2, logo_size*2)
    asp_ratio = min(target_w / w, target_h / h)
    new_w = int(w * asp_ratio)
    new_h = int(h * asp_ratio)

    logo_resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logo_pad = cv2.copyMakeBorder(
        logo_resized, 
        (target_h - new_h) // 2, 
        (target_h - new_h) - ((target_h - new_h) // 2), 
        (target_w - new_w) // 2, 
        (target_w - new_w) - ((target_w - new_w) // 2), 
        cv2.BORDER_CONSTANT, 
        value=pad_color
    )

    # Crop to the same vertical profile used by card rendering.
    logo_crop = logo_pad[logo_size//2-(CARD_H-logo_size)//2:logo_size+logo_size//2+(CARD_H-logo_size)//2, :logo_size*2]

    if logo_crop.shape[2] == 4:
        alpha = logo_crop[:,:,3] / 255.0
        for c in range(3):
            canvas[y-(CARD_H-logo_size)//2:y+logo_size+(CARD_H-logo_size)//2, x:x+logo_size*2, c] = (
                alpha * logo_crop[:,:,c] +
                (1 - alpha) * canvas[y-(CARD_H-logo_size)//2:y+logo_size+(CARD_H-logo_size)//2, x:x+logo_size*2, c]
            )
    else:
        canvas[y-(CARD_H-logo_size)//2:y+logo_size+(CARD_H-logo_size)//2, x:x+logo_size*2] = logo_crop


def overlay_image(canvas, path, x, y, image_size, pad_color=(0, 0, 0)):
    """Overlay any square-fitted image (cup art, champion logo, etc.)."""
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return
    
    h, w = logo.shape[:2]
    target_h, target_w = (image_size, image_size)
    asp_ratio = min(target_w / w, target_h / h)
    new_w = int(w * asp_ratio)
    new_h = int(h * asp_ratio)

    logo_resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logo_pad = cv2.copyMakeBorder(
        logo_resized, 
        (target_h - new_h) // 2, 
        (target_h - new_h) - ((target_h - new_h) // 2), 
        (target_w - new_w) // 2, 
        (target_w - new_w) - ((target_w - new_w) // 2), 
        cv2.BORDER_CONSTANT, 
        value=pad_color
    )

    if logo_pad.shape[2] == 4:
        alpha = logo_pad[:,:,3] / 255.0
        for c in range(3):
            canvas[y:y+image_size, x:x+image_size, c] = (
                alpha * logo_pad[:,:,c] +
                (1 - alpha) * canvas[y:y+image_size, x:x+image_size, c]
            )
    else:
        canvas[y:y+image_size, x:x+image_size] = logo_pad[:,:,:3]


def choose_card(round1):
    """Return the card dimensions for the given playoff round type."""
    if round1:
        return CARD_1_W, CARD_1_H
    else:
        return CARD_W, CARD_H


def draw_card(canvas, x, y, team, seed=None, round1=False, winner=False, align_right=False):
    """Draw one team card with logo, optional seed, and winner highlight.

    Args:
        align_right: If True, x is treated as the card's right edge so cards on
            the right side of the bracket mirror left-side placement.
    """

    card_w, card_h = choose_card(round1)

    if align_right:
        x -= card_w

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x,y), (x+card_w,y+card_h), (40,40,40), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    # Round 1 cards reserve a black strip for seed text and mirror by side.
    if round1 and align_right:
        cv2.rectangle(canvas, (x,y), (x+card_w//3*2,y+card_h), cons.team_info[team]['c1'], -1)
        cv2.rectangle(canvas, (x+card_w//3*2,y), (x+card_w,y+card_h), (0,0,0), -1)
    elif round1:
        cv2.rectangle(canvas, (x,y), (x+card_w//3,y+card_h), (0,0,0), -1)
        cv2.rectangle(canvas, (x+card_w//3,y), (x+card_w,y+card_h), cons.team_info[team]['c1'], -1)
    else:
        cv2.rectangle(canvas, (x,y), (x+card_w,y+card_h), cons.team_info[team]['c1'], -1)

    # Winner cards get an accent border so advancement is visually obvious.
    border = (0,180,255) if winner else (70,70,70)
    cv2.rectangle(canvas, (x,y), (x+card_w,y+card_h), border, 2)

    if round1:
        lx = x + 50 if not align_right else x + card_w - LOGO_SIZE_CARD - 100
    else:
        lx = x if not align_right else x + card_w - LOGO_SIZE_CARD - 50
    overlay_logo(canvas, cons.team_info[team]['logo'], lx, y+7, LOGO_SIZE_CARD, pad_color=cons.team_info[team]['c1'])

    if round1: 
        text = f'{seed}'
        offset = 2 if len(seed) == 3 else 12
        tx = x + offset if not align_right else x + 100 + offset

        cv2.putText(canvas, text, (tx, y+40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (240,240,240), 1, cv2.LINE_AA)


def connect_left(canvas, x, y1, y2, score, x_offset, round1=False):
    """Connect two left-side cards to the next-round spine and draw score."""

    card_w, card_h = choose_card(round1)

    mid_y = (y1 + y2)//2
    lx = x + card_w + 30 if not round1 else x + card_w + 90

    draw_glow_line(canvas, (x+card_w, y1+card_h//2), (lx, y1+card_h//2))
    draw_glow_line(canvas, (x+card_w, y2+card_h//2), (lx, y2+card_h//2))
    draw_glow_line(canvas, (lx, y1+card_h//2), (lx, y2+card_h//2))

    draw_series_score(canvas, lx-x_offset, mid_y+card_h//2, score)
    return mid_y


def connect_right(canvas, x, y1, y2, score, x_offset, round1=False):
    """Connect two right-side cards to the next-round spine and draw score."""
    
    card_w, card_h = choose_card(round1)

    mid_y = (y1 + y2)//2
    lx = x - card_w - 30 if not round1 else x - card_w - 90

    draw_glow_line(canvas, (x-card_w, y1+card_h//2), (lx, y1+card_h//2))
    draw_glow_line(canvas, (x-card_w, y2+card_h//2), (lx, y2+card_h//2))
    draw_glow_line(canvas, (lx, y1+card_h//2), (lx, y2+card_h//2))

    draw_series_score(canvas, lx+x_offset, mid_y+card_h//2, score)
    return mid_y


def get_round_positions(start, spacing, n):
    """Build evenly spaced y-positions for card rows in a round."""
    return [start + i*spacing for i in range(n)]


def display_playoff_tree(matchups, season, pred_date, display_image=True):
    """Render the playoff bracket image from simulated matchup outcomes.

    Bracket structure expectation:
    - matchups[1]: 8 first-round series (0-3 East, 4-7 West)
    - matchups[2]: 4 second-round series (0-1 East, 2-3 West)
    - matchups[3]: 2 conference finals (0 East, 1 West)
    - matchups[4][0]: Stanley Cup Final
    """

    # Precompute y-lanes for each round so cards and connectors align cleanly.
    r1_y = get_round_positions(BASE_Y, R1_SPACE, 8)
    r2_y = get_round_positions(BASE_Y + R1_SPACE//2, R2_SPACE, 4)
    r3_y = get_round_positions(BASE_Y + R1_SPACE//2 + R2_SPACE//2, R3_SPACE, 2)
    final_y = get_round_positions(BASE_Y + R1_SPACE//2 + R2_SPACE//2 + R3_SPACE//2, R3_SPACE*2, 1)

    # Center header: cup art and final champion logo.
    overlay_image(CANVAS, cons.images_folder + cons.stanley_cup_image, CENTER_X-LOGO_SIZE_CUP//2, 250, LOGO_SIZE_CUP, pad_color=(0,0,0))
    cv2.putText(CANVAS, "STANLEY CUP CHAMPION",
                (CENTER_X-230, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (240,240,240), 2)
    
    cup_champ = matchups[4][0].get_series_winner()
    overlay_image(CANVAS, cons.team_info[cup_champ]['logo'], CENTER_X-LOGO_SIZE_CHAMP//2, 100, LOGO_SIZE_CHAMP, pad_color=cons.team_info[cup_champ]['c1'])

    # Draw Western Conference on the left side of the bracket.
    # West round 1
    for i in range(4, 8):
        y1, y2 = r1_y[(i-4)*2], r1_y[(i-4)*2+1]
        t1 = matchups[1][i].get_team1()
        seed1 = matchups[1][i].get_playoff_seed(t1)
        t2 = matchups[1][i].get_team2()
        seed2 = matchups[1][i].get_playoff_seed(t2)
        score = matchups[1][i].get_series_score()
        winner = matchups[1][i].get_series_winner()

        draw_card(CANVAS, LEFT_X[0], y1, t1, seed1, round1=True, winner=(winner==t1))
        draw_card(CANVAS, LEFT_X[0], y2, t2, seed2, round1=True, winner=(winner==t2))
        connect_left(CANVAS, LEFT_X[0], y1, y2, score, 45, round1=True)

    # West round 2 uses matchup indices 2 and 3 by bracket convention.
    for i in range(2, 4):
        y = r2_y[(i-2)*2]
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()
        winner = matchups[2][i].get_series_winner()

        draw_card(CANVAS, LEFT_X[1], y, t1, winner=(winner==t1))
        draw_card(CANVAS, LEFT_X[1], y+R2_SPACE, t2, winner=(winner==t2))

        connect_left(CANVAS, LEFT_X[1], y, y+R2_SPACE, score, 45)

    # West conference final is matchup index 1 in round 3.
    for i in range(2):
        y = r3_y[(i-1)*2]
        t1 = matchups[3][i].get_team1()
        t2 = matchups[3][i].get_team2()
        score = matchups[3][i].get_series_score()
        winner = matchups[3][i].get_series_winner()
        west_final_winner = winner

        draw_card(CANVAS, LEFT_X[2], y, t1, winner=(winner==t1))
        draw_card(CANVAS, LEFT_X[2], y+R3_SPACE, t2, winner=(winner==t2))
        connect_left(CANVAS, LEFT_X[2], y, y+R3_SPACE, score, 45)

    # Draw Eastern Conference on the right side, mirrored with align_right=True.
    # East round 1
    for i in range(0, 4):
        y1, y2 = r1_y[i*2], r1_y[i*2+1]
        t1 = matchups[1][i].get_team1()
        seed1 = matchups[1][i].get_playoff_seed(t1)
        t2 = matchups[1][i].get_team2()
        seed2 = matchups[1][i].get_playoff_seed(t2)
        score = matchups[1][i].get_series_score()
        winner = matchups[1][i].get_series_winner()

        draw_card(CANVAS, RIGHT_X[0], y1, t1, seed1, round1=True, winner=(winner==t1), align_right=True)
        draw_card(CANVAS, RIGHT_X[0], y2, t2, seed2, round1=True, winner=(winner==t2), align_right=True)

        connect_right(CANVAS, RIGHT_X[0], y1, y2, score, 45, round1=True)

    # East round 2 uses matchup indices 0 and 1.
    for i in range(2):
        y = r2_y[i*2]
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()
        winner = matchups[2][i].get_series_winner()

        draw_card(CANVAS, RIGHT_X[1], y, t1, winner=(winner==t1), align_right=True)
        draw_card(CANVAS, RIGHT_X[1], y+R2_SPACE, t2, winner=(winner==t2), align_right=True)

        connect_right(CANVAS, RIGHT_X[1], y, y+R2_SPACE, score, 45)

    # East conference final is matchup index 0 in round 3.
    for i in range(1):
        y = r3_y[i*2]
        t1 = matchups[3][i].get_team1()
        t2 = matchups[3][i].get_team2()
        score = matchups[3][i].get_series_score()
        winner = matchups[3][i].get_series_winner()
        east_final_winner = winner

        draw_card(CANVAS, RIGHT_X[2], y, t1, winner=(winner==t1), align_right=True)
        draw_card(CANVAS, RIGHT_X[2], y+R3_SPACE, t2, winner=(winner==t2), align_right=True)

        connect_right(CANVAS, RIGHT_X[2], y, y+R3_SPACE, score, 45)

    # Stanley Cup Final card pair and center score badge.
    if west_final_winner == cup_champ:
        west_win, east_win = True, False
    elif east_final_winner == cup_champ:
        west_win, east_win = False, True
    draw_card(CANVAS, CENTER_X-150, final_y[0], west_final_winner, winner=west_win)
    draw_card(CANVAS, CENTER_X+150, final_y[0], east_final_winner, winner=east_win, align_right=True)

    draw_series_score(CANVAS, CENTER_X, final_y[0] + CARD_H//2, matchups[4][0].get_series_score())

    # Display the image, save it, and wait for key press to close
    print(f'Saving playoff tree graphic...\n')
    cv2.imwrite(f'{cons.season_pred_folder.format(date=pred_date)}{cons.playoff_tree_filename.format(season=season, date=pred_date)}', CANVAS)
    
    if display_image:
        cv2.imshow(f"NHL Playoff Bracket {season}", CANVAS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()