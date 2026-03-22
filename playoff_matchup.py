class PlayoffMatchup:
    def __init__(self, team1, team2, team1_conf_seed, team2_conf_seed, pl_round, conference=None, division=None):
        self.team1 = team1
        self.team2 = team2
        self.team1_conf_seed = team1_conf_seed
        self.team2_conf_seed = team2_conf_seed
        self.conference = conference
        self.division = division
        self.pl_round = pl_round
        self.series_winner = None
        self.series_loser = None

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

    def set_series_results(self, winner, loser):
        self.series_winner = winner
        self.series_loser = loser
    
    def get_series_winner(self):
        return self.series_winner
    
    def get_series_loser(self):
        return self.series_loser