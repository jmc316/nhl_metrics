class PlayoffMatchup:
    def __init__(self, team1, team2, team1_conf_seed, team2_conf_seed, pl_round, team1_playoff_seed=None, team_2_playoff_seed=None, conference=None, division=None):
        self.team1 = team1
        self.team2 = team2
        self.team1_conf_seed = int(team1_conf_seed)
        self.team2_conf_seed = int(team2_conf_seed)
        self.team1_playoff_seed = team1_playoff_seed
        self.team2_playoff_seed = team_2_playoff_seed
        self.conference = conference
        self.division = division
        self.pl_round = pl_round
        self.series_winner = None
        self.series_loser = None
        self.series_winner_score = None
        self.series_loser_score = None

    def get_winner_conf_seed(self):
        if self.series_winner == self.team1:
            return int(self.team1_conf_seed)
        elif self.series_winner == self.team2:
            return int(self.team2_conf_seed)
        else:
            return None

    def get_conference(self):
        return self.conference
    
    def get_division(self):
        return self.division

    def get_team1(self):
        return self.team1
    
    def get_team2(self):
        return self.team2

    def get_teams(self):
        return self.team1, self.team2

    def set_series_results(self, winner, loser, loser_score):
        self.series_winner = winner
        self.series_loser = loser
        self.series_winner_score = 4
        self.series_loser_score = int(loser_score)

    def get_series_winner(self):
        return self.series_winner
    
    def get_series_loser(self):
        return self.series_loser
    
    def get_series_score(self):
        return str(self.series_winner_score) + '-' + str(self.series_loser_score)
    
    def get_playoff_seed(self, team):
        if team == self.team1:
            return self.team1_playoff_seed
        elif team == self.team2:
            return self.team2_playoff_seed
        else:
            return None