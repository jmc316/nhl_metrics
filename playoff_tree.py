from importlib.resources import path

import cv2

import numpy as np
import constants as cons

from playoff_matchup import PlayoffMatchup

# ---------------- CANVAS ----------------
WIDTH, HEIGHT = 1400, 800
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
canvas[:] = (12, 12, 14)

left_x = [20, 270, 410]
right_x = [WIDTH-left_x[0], WIDTH-left_x[1], WIDTH-left_x[2]]
center_x = WIDTH // 2

card_1_w, card_1_h = 150, 65
card_w, card_h = 100, 65
logo_size_card = 50
logo_size_champ = 150
logo_size_cup = 500

# ---------------- HELPERS ----------------
def draw_glow_line(canvas, pt1, pt2, color=(0,180,255)):
    overlay = canvas.copy()
    cv2.line(overlay, pt1, pt2, color, 6)
    canvas[:] = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    cv2.line(canvas, pt1, pt2, (240,240,240), 2)


def draw_series_score(canvas, x, y, text):
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
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return
    
    h, w = logo.shape[:2]
    target_h, target_w = (logo_size, logo_size)
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
            canvas[y:y+logo_size, x:x+logo_size, c] = (
                alpha * logo_pad[:,:,c] +
                (1 - alpha) * canvas[y:y+logo_size, x:x+logo_size, c]
            )
    else:
        canvas[y:y+logo_size, x:x+logo_size] = logo_pad


def draw_card(canvas, x, y, team, seed=None, round1=False, winner=False, align_right=False):

    if round1:
        card_w_loc, card_h_loc = card_1_w, card_1_h
    else:
        card_w_loc, card_h_loc = card_w, card_h

    if align_right:
        x -= card_w_loc

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x,y), (x+card_w_loc,y+card_h_loc), (40,40,40), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    if round1 and align_right:
        cv2.rectangle(canvas, (x,y), (x+card_w_loc//3*2,y+card_h), cons.team_info[team]['c1'], -1)
        cv2.rectangle(canvas, (x+card_w_loc//3*2,y), (x+card_w_loc,y+card_h), (0,0,0), -1)
    elif round1:
        cv2.rectangle(canvas, (x,y), (x+card_w_loc//3,y+card_h), (0,0,0), -1)
        cv2.rectangle(canvas, (x+card_w_loc//3,y), (x+card_w_loc,y+card_h), cons.team_info[team]['c1'], -1)
    else:
        cv2.rectangle(canvas, (x,y), (x+card_w_loc,y+card_h), cons.team_info[team]['c1'], -1)

    border = (0,180,255) if winner else (70,70,70)
    cv2.rectangle(canvas, (x,y), (x+card_w_loc,y+card_h_loc), border, 1)

    if round1:
        lx = x + 75 if not align_right else x + card_w_loc - logo_size_card - 75
    else:
        lx = x + 25 if not align_right else x + card_w_loc - logo_size_card - 25
    overlay_logo(canvas, cons.team_info[team]['logo'], lx, y+7, logo_size_card, pad_color=cons.team_info[team]['c1'])

    if round1: 
        text = f'{seed}'
        offset = 2 if len(seed) == 3 else 12
        tx = x + offset if not align_right else x + 100 + offset

        cv2.putText(canvas, text, (tx, y+40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (240,240,240), 1, cv2.LINE_AA)


def connect_left(canvas, x, y1, y2, score, x_offset, round1=False):

    if round1:
        card_w_loc, card_h_loc = card_1_w, card_1_h
    else:
        card_w_loc, card_h_loc = card_w, card_h

    mid_y = (y1 + y2)//2
    lx = x + card_w_loc + 30 if not round1 else x + card_w_loc + 90

    draw_glow_line(canvas, (x+card_w_loc, y1+card_h_loc//2), (lx, y1+card_h_loc//2))
    draw_glow_line(canvas, (x+card_w_loc, y2+card_h_loc//2), (lx, y2+card_h_loc//2))
    draw_glow_line(canvas, (lx, y1+card_h_loc//2), (lx, y2+card_h_loc//2))

    draw_series_score(canvas, lx-x_offset, mid_y+card_h_loc//2, score)
    return mid_y


def connect_right(canvas, x, y1, y2, score, x_offset, round1=False):
    if round1:
        card_w_loc, card_h_loc = card_1_w, card_1_h
    else:
        card_w_loc, card_h_loc = card_w, card_h

    mid_y = (y1 + y2)//2
    lx = x - card_w_loc - 30 if not round1 else x - card_w_loc - 90

    draw_glow_line(canvas, (x-card_w_loc, y1+card_h_loc//2), (lx, y1+card_h_loc//2))
    draw_glow_line(canvas, (x-card_w_loc, y2+card_h_loc//2), (lx, y2+card_h_loc//2))
    draw_glow_line(canvas, (lx, y1+card_h_loc//2), (lx, y2+card_h_loc//2))

    draw_series_score(canvas, lx+x_offset, mid_y+card_h_loc//2, score)
    return mid_y


def get_round_positions(start, spacing, n):
    return [start + i*spacing for i in range(n)]


# ---------------- SPACING ----------------
base_y = 150
r1_space = 80
r2_space = r1_space * 2
r3_space = r2_space * 2

r1_y = get_round_positions(base_y, r1_space, 8)
r2_y = get_round_positions(base_y + r1_space//2, r2_space, 4)
r3_y = get_round_positions(base_y + r1_space//2 + r2_space//2, r3_space, 2)
final_y = get_round_positions(base_y + r1_space//2 + r2_space//2 + r3_space//2, r3_space*2, 1)


def display_playoff_tree(matchups, season, pred_date):

    # ---------------- TITLE ----------------
    overlay_logo(canvas, 'images/stanley_cup.png', center_x-logo_size_cup//2, 250, logo_size_cup, pad_color=(0,0,0))
    cv2.putText(canvas, "STANLEY CUP CHAMPION",
                (center_x-230, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (240,240,240), 2)
    
    cup_champ = matchups[4][0].get_series_winner()
    overlay_logo(canvas, cons.team_info[cup_champ]['logo'], center_x-logo_size_champ//2, 100, logo_size_champ, pad_color=cons.team_info[cup_champ]['c1'])

    # ---------------- WEST ----------------
    wy = []
    west_r1_winners = []

    for i in range(4, 8):
        y1, y2 = r1_y[(i-4)*2], r1_y[(i-4)*2+1]
        t1 = matchups[1][i].get_team1()
        seed1 = matchups[1][i].get_playoff_seed(t1)
        t2 = matchups[1][i].get_team2()
        seed2 = matchups[1][i].get_playoff_seed(t2)
        score = matchups[1][i].get_series_score()

        winner = matchups[1][i].get_series_winner()
        west_r1_winners.append(winner)

        draw_card(canvas, left_x[0], y1, t1, seed1, round1=True, winner=(winner==t1))
        draw_card(canvas, left_x[0], y2, t2, seed2, round1=True, winner=(winner==t2))

        wy.append(connect_left(canvas, left_x[0], y1, y2, score, 45, round1=True))

    wy2 = []
    west_r2_winners = []

    for i in range(2, 4):
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()

        winner = matchups[2][i].get_series_winner()
        west_r2_winners.append(winner)

        y = r2_y[(i-2)*2]
        draw_card(canvas, left_x[1], y, t1, winner=(winner==t1))
        draw_card(canvas, left_x[1], y+r2_space, t2, winner=(winner==t2))

        wy2.append(connect_left(canvas, left_x[1], y, y+r2_space, score, 45))

    wy3 = []
    for i in range(2):
        t1 = matchups[3][i].get_team1()
        t2 = matchups[3][i].get_team2()
        score = matchups[3][i].get_series_score()

        winner = matchups[3][i].get_series_winner()
        west_final_winner = winner

        y = r3_y[(i-1)*2]
        draw_card(canvas, left_x[2], y, t1, winner=(winner==t1))
        draw_card(canvas, left_x[2], y+r3_space, t2, winner=(winner==t2))

        wy3.append(connect_left(canvas, left_x[2], y, y+r3_space, score, 45))

    # ---------------- EAST ----------------
    ey = []
    east_r1_winners = []

    for i in range(0, 4):
        y1, y2 = r1_y[i*2], r1_y[i*2+1]
        t1 = matchups[1][i].get_team1()
        seed1 = matchups[1][i].get_playoff_seed(t1)
        t2 = matchups[1][i].get_team2()
        seed2 = matchups[1][i].get_playoff_seed(t2)
        score = matchups[1][i].get_series_score()

        winner = matchups[1][i].get_series_winner()
        east_r1_winners.append(winner)

        draw_card(canvas, right_x[0], y1, t1, seed1, round1=True, winner=(winner==t1), align_right=True)
        draw_card(canvas, right_x[0], y2, t2, seed2, round1=True, winner=(winner==t2), align_right=True)

        ey.append(connect_right(canvas, right_x[0], y1, y2, score, 45, round1=True))

    ey2 = []
    east_r2_winners = []

    for i in range(2):
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()

        winner = matchups[2][i].get_series_winner()
        east_r2_winners.append(winner)

        y = r2_y[i*2]
        draw_card(canvas, right_x[1], y, t1, winner=(winner==t1), align_right=True)
        draw_card(canvas, right_x[1], y+r2_space, t2, winner=(winner==t2), align_right=True)

        ey2.append(connect_right(canvas, right_x[1], y, y+r2_space, score, 45))

    ey3 = []
    for i in range(1):
        t1 = matchups[3][i].get_team1()
        t2 = matchups[3][i].get_team2()
        score = matchups[3][i].get_series_score()

        winner = matchups[3][i].get_series_winner()
        east_final_winner = winner

        y = r3_y[i*2]
        draw_card(canvas, right_x[2], y+r3_space, t1, winner=(winner==t1), align_right=True)
        draw_card(canvas, right_x[2], y, t2, winner=(winner==t2), align_right=True)

        ey3.append(connect_right(canvas, right_x[2], y, y+r3_space, score, 45))

    # ---------------- FINAL ----------------
    if west_final_winner == cup_champ:
        west_win, east_win = True, False
    elif east_final_winner == cup_champ:
        west_win, east_win = False, True
    draw_card(canvas, center_x-150, final_y[0], west_final_winner, winner=west_win)
    draw_card(canvas, center_x+150, final_y[0], east_final_winner, winner=east_win, align_right=True)

    draw_series_score(canvas, center_x, final_y[0] + card_h//2, matchups[4][0].get_series_score())

    # ---------------- OUTPUT ----------------
    cv2.imwrite(f'{cons.season_pred_folder.format(date=pred_date)}{season}_playoff_tree_{pred_date}.png', canvas)
    cv2.imshow(f"NHL Playoff Bracket {season}", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # create some dummy playoff matchups for testing
    playoff_matchups = {}

    display_playoff_tree(playoff_matchups, '20252026', '2026-03-24')