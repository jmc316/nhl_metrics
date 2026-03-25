import cv2

import numpy as np
import constants as cons

# local constants
WIDTH, HEIGHT = 1400, 800
CANVAS = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
CANVAS[:] = (12, 12, 14)
LEFT_X = [20, 270, 410]
RIGHT_X = [WIDTH-LEFT_X[0], WIDTH-LEFT_X[1], WIDTH-LEFT_X[2]]
CENTER_X = WIDTH // 2
CARD_1_W, CARD_1_H = 150, 65
CARD_W, CARD_H = 100, 65
LOGO_SIZE_CARD = 50
LOGO_SIZE_CHAMP = 150
LOGO_SIZE_CUP = 500
BASE_Y = 150
R1_SPACE = 80
R2_SPACE = R1_SPACE * 2
R3_SPACE = R2_SPACE * 2


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


def choose_card(round1):
    if round1:
        return CARD_1_W, CARD_1_H
    else:
        return CARD_W, CARD_H


def draw_card(canvas, x, y, team, seed=None, round1=False, winner=False, align_right=False):

    card_w, card_h = choose_card(round1)

    if align_right:
        x -= card_w

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x,y), (x+card_w,y+card_h), (40,40,40), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    if round1 and align_right:
        cv2.rectangle(canvas, (x,y), (x+card_w//3*2,y+card_h), cons.team_info[team]['c1'], -1)
        cv2.rectangle(canvas, (x+card_w//3*2,y), (x+card_w,y+card_h), (0,0,0), -1)
    elif round1:
        cv2.rectangle(canvas, (x,y), (x+card_w//3,y+card_h), (0,0,0), -1)
        cv2.rectangle(canvas, (x+card_w//3,y), (x+card_w,y+card_h), cons.team_info[team]['c1'], -1)
    else:
        cv2.rectangle(canvas, (x,y), (x+card_w,y+card_h), cons.team_info[team]['c1'], -1)

    border = (0,180,255) if winner else (70,70,70)
    cv2.rectangle(canvas, (x,y), (x+card_w,y+card_h), border, 1)

    if round1:
        lx = x + 75 if not align_right else x + card_w - LOGO_SIZE_CARD - 75
    else:
        lx = x + 25 if not align_right else x + card_w - LOGO_SIZE_CARD - 25
    overlay_logo(canvas, cons.team_info[team]['logo'], lx, y+7, LOGO_SIZE_CARD, pad_color=cons.team_info[team]['c1'])

    if round1: 
        text = f'{seed}'
        offset = 2 if len(seed) == 3 else 12
        tx = x + offset if not align_right else x + 100 + offset

        cv2.putText(canvas, text, (tx, y+40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (240,240,240), 1, cv2.LINE_AA)


def connect_left(canvas, x, y1, y2, score, x_offset, round1=False):

    card_w, card_h = choose_card(round1)

    mid_y = (y1 + y2)//2
    lx = x + card_w + 30 if not round1 else x + card_w + 90

    draw_glow_line(canvas, (x+card_w, y1+card_h//2), (lx, y1+card_h//2))
    draw_glow_line(canvas, (x+card_w, y2+card_h//2), (lx, y2+card_h//2))
    draw_glow_line(canvas, (lx, y1+card_h//2), (lx, y2+card_h//2))

    draw_series_score(canvas, lx-x_offset, mid_y+card_h//2, score)
    return mid_y


def connect_right(canvas, x, y1, y2, score, x_offset, round1=False):
    
    card_w, card_h = choose_card(round1)

    mid_y = (y1 + y2)//2
    lx = x - card_w - 30 if not round1 else x - card_w - 90

    draw_glow_line(canvas, (x-card_w, y1+card_h//2), (lx, y1+card_h//2))
    draw_glow_line(canvas, (x-card_w, y2+card_h//2), (lx, y2+card_h//2))
    draw_glow_line(canvas, (lx, y1+card_h//2), (lx, y2+card_h//2))

    draw_series_score(canvas, lx+x_offset, mid_y+card_h//2, score)
    return mid_y


def get_round_positions(start, spacing, n):
    return [start + i*spacing for i in range(n)]


def display_playoff_tree(matchups, season, pred_date):

    r1_y = get_round_positions(BASE_Y, R1_SPACE, 8)
    r2_y = get_round_positions(BASE_Y + R1_SPACE//2, R2_SPACE, 4)
    r3_y = get_round_positions(BASE_Y + R1_SPACE//2 + R2_SPACE//2, R3_SPACE, 2)
    final_y = get_round_positions(BASE_Y + R1_SPACE//2 + R2_SPACE//2 + R3_SPACE//2, R3_SPACE*2, 1)

    # print stanley cup champion and stanley gup logo
    overlay_logo(CANVAS, 'images/stanley_cup.png', CENTER_X-LOGO_SIZE_CUP//2, 250, LOGO_SIZE_CUP, pad_color=(0,0,0))
    cv2.putText(CANVAS, "STANLEY CUP CHAMPION",
                (CENTER_X-230, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (240,240,240), 2)
    
    cup_champ = matchups[4][0].get_series_winner()
    overlay_logo(CANVAS, cons.team_info[cup_champ]['logo'], CENTER_X-LOGO_SIZE_CHAMP//2, 100, LOGO_SIZE_CHAMP, pad_color=cons.team_info[cup_champ]['c1'])

    # draw the Western Conference Matchups on the left
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

    # West round 2
    for i in range(2, 4):
        y = r2_y[(i-2)*2]
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()
        winner = matchups[2][i].get_series_winner()

        draw_card(CANVAS, LEFT_X[1], y, t1, winner=(winner==t1))
        draw_card(CANVAS, LEFT_X[1], y+R2_SPACE, t2, winner=(winner==t2))

        connect_left(CANVAS, LEFT_X[1], y, y+R2_SPACE, score, 45)

    # West round 3
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

    # draw the Eastern Conference Matchups on the right
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

    # East round 2
    for i in range(2):
        y = r2_y[i*2]
        t1 = matchups[2][i].get_team1()
        t2 = matchups[2][i].get_team2()
        score = matchups[2][i].get_series_score()
        winner = matchups[2][i].get_series_winner()

        draw_card(CANVAS, RIGHT_X[1], y, t1, winner=(winner==t1), align_right=True)
        draw_card(CANVAS, RIGHT_X[1], y+R2_SPACE, t2, winner=(winner==t2), align_right=True)

        connect_right(CANVAS, RIGHT_X[1], y, y+R2_SPACE, score, 45)

    # East round 3
    for i in range(1):
        y = r3_y[i*2]
        t1 = matchups[3][i].get_team1()
        t2 = matchups[3][i].get_team2()
        score = matchups[3][i].get_series_score()
        winner = matchups[3][i].get_series_winner()
        east_final_winner = winner

        draw_card(CANVAS, RIGHT_X[2], y+R3_SPACE, t1, winner=(winner==t1), align_right=True)
        draw_card(CANVAS, RIGHT_X[2], y, t2, winner=(winner==t2), align_right=True)

        connect_right(CANVAS, RIGHT_X[2], y, y+R3_SPACE, score, 45)

    # Stanley Cup Final
    if west_final_winner == cup_champ:
        west_win, east_win = True, False
    elif east_final_winner == cup_champ:
        west_win, east_win = False, True
    draw_card(CANVAS, CENTER_X-150, final_y[0], west_final_winner, winner=west_win)
    draw_card(CANVAS, CENTER_X+150, final_y[0], east_final_winner, winner=east_win, align_right=True)

    draw_series_score(CANVAS, CENTER_X, final_y[0] + CARD_H//2, matchups[4][0].get_series_score())

    # Display the image, save it, and wait for key press to close
    cv2.imwrite(f'{cons.season_pred_folder.format(date=pred_date)}{cons.playoff_tree_filename.format(season=season, date=pred_date)}', CANVAS)
    cv2.imshow(f"NHL Playoff Bracket {season}", CANVAS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()