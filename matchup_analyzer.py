import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, commonteamroster, playergamelog

# =============
# CONFIGURATION
# =============

@dataclass
class Config:
  """Configuration settings for the analyzer"""
  current_season: str = '2025-26'
  rate_limit_delay: float = 0.01
  position_order: List[str] = field(default_factory=lambda: ['PG', 'SG', 'SF', 'PF', 'C', 'UNKNOWN'])

  @staticmethod
  def normalize_position(position: str) -> str:
    """Normalize position strings to standard 5 positions"""
    if pd.isna(position) or position == '':
      return 'UNKNOWN'

    pos = str(position).upper().strip()

    # Specific positions
    if 'POINT' in pos or pos == 'PG':
      return 'PG'
    if 'SHOOTING' in pos or pos == 'SG':
      return 'SG'
    if 'SMALL' in pos or pos == 'SF':
      return 'SF'
    if 'POWER' in pos or pos == 'PF':
      return 'PF'
    if 'CENTER' in pos or pos == 'C' or pos == 'CENTER':
      return 'C'

    # Hybrid positions
    if ('G' in pos and 'F' in pos):
      return 'SF'
    if ('F' in pos and 'C' in pos):
      return 'PF'

    # Generic
    if 'G' in pos or 'GUARD' in pos:
      return 'SG'
    if 'F' in pos or 'FORWARD' in pos:
      return 'SF'

    return 'UNKNOWN'

# ===========
# DATA MODELS
# ===========

@dataclass
class DefensiveStats:
  """Defensive statistics by position for a specific time window"""
  position: str
  total_points: float
  avg_per_game: float
  games_analyzed: int

  def to_dict(self) -> Dict:
    """Convert to dictionary for easy serialization"""
    return {
      'Position': self.position,
      'Total Points': self.total_points,
      'Avg_Per_Game': self.avg_per_game,
      'Games': self.games_analyzed
    }

@dataclass
class DefensiveProfile:
  """Complete defensive profile across multiple time windows"""
  last_7: List[DefensiveStats] = field(default_factory=list)
  last_15: List[DefensiveStats] = field(default_factory=list)
  season: List[DefensiveStats] = field(default_factory=list)

  def get_dataframe(self, window: str = 'season') -> pd.DataFrame:
    """Get defensive stats as DataFrame for specific window"""
    data_map = {
      '7': self.last_7,
      '15': self.last_15,
      'season': self.season
    }

    stats = data_map.get(window, self.season)
    if not stats:
      return pd.DataFrame()

    return pd.DataFrame([s.to_dict() for s in stats])

  def get_weakest_positions(self, window:str = 'season', n: int = 2) -> List[DefensiveStats]:
    """Get the N weakest defensive positions"""
    data_map = {
      '7': self.last_7,
      '15': self.last_15,
      'season': self.season
    }

    stats = data_map.get(window, self.season)
    if not stats:
      return []

    sorted_stats = sorted(stats, key=lambda x: x.avg_per_game, reverse=True)
    return sorted_stats[:n]

  def get_position_avg(self, position: str, window: str = 'season') -> Optional[float]:
    """Get average points allowed to a specific position"""
    data_map = {
      '7': self.last_7,
      '15': self.last_15,
      'season': self.season
    }

    stats = data_map.get(window, self.season)
    for stat in stats:
      if stat.position == position:
        return stat.avg_per_game
    return None

@dataclass
class PlayerStats:
  """Individual player stats"""
  player_id: str
  name: str
  jersey: str
  position: str
  listed_position: str
  ppg_last_7: float
  ppg_last_15: float
  ppg_season: float
  games_played: int

  @property
  def is_hot(self) -> bool:
    """Check if player is scoring above season average"""
    return self.ppg_last_7 > (self.ppg_season * 1.15)

  @property
  def is_cold(self) -> bool:
    """Check if player is scoring below season average"""
    return self.ppg_last_7 < (self.ppg_season * 0.85)

  @property
  def form_differential(self) -> float:
    """Difference between recent form and season average"""
    return self.ppg_last_7 - self.ppg_season

  def to_dict(self) -> Dict:
    """Convert to dictionary"""
    return {
      'Player': self.name,
      'Jersey': self.jersey,
      'Position': self.position,
      'Listed_Position': self.listed_position,
      'PPG_L7': self.ppg_last_7,
      'PPG_L15': self.ppg_last_15,
      'PPG_Season': self.ppg_season,
      'GP': self.games_played,
      'Form': self.form_differential
    }

@dataclass
class TeamAnalysis:
  """Complete analysis for one team"""
  team_id: int
  team_abbrev: str
  team_name: str
  defense: DefensiveProfile = field(default_factory=DefensiveProfile)
  players: List[PlayerStats] = field(default_factory=list)

  def get_players_dataframe(self) -> pd.DataFrame:
    """Get all players as DataFrame"""
    if not self.players:
      return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in self.players])

  def get_players_by_position(self, position: str) -> List[PlayerStats]:
    """Get all players at a specific position"""
    return [p for p in self.players if p.position == position]

  def get_top_scorers(self, n: int = 5) -> List[PlayerStats]:
    """Get top N scorers by season average"""
    sorted_players = sorted(self.players, key=lambda x: x.ppg_season, reverse=True)
    return sorted_players[:n]

  def get_hot_players(self) -> List[PlayerStats]:
    """Get all players currently in hot form"""
    return [p for p in self.players if p.is_hot]

  def get_cold_players(self) -> List[PlayerStats]:
    """Get all players currently in cold form"""
    return [p for p in self.players if p.is_cold]

@dataclass
class MatchupInsight:
  "A single matchup insight"
  offensive_player: PlayerStats
  defensive_position: DefensiveStats
  matchup_score: float # How favorable the matchup is

  @property
  def description(self) -> str:
    """Get human-readable description."""
    return (
      f"{self.offensive_player.position} Matchup: "
      f"{self.offensive_player.name} ({self.offensive_player.ppg_season:.1f} PPG) "
      f"vs defense allowing {self.defensive_position.avg_per_game:.1f} PPG"
    )

  @property
  def is_favorable(self) -> bool:
    "Check if matchup is favorable for offense."
    return self.matchup_score > 1.0

@dataclass
class MatchupAnalysis:
  "A Complete matchup analysis between two teams"
  team1: TeamAnalysis
  team2: TeamAnalysis
  team1_insights: List[MatchupInsight] = field(default_factory=list)
  team2_insights: List[MatchupInsight] = field(default_factory=list)

  def get_defense_comparison(self, window: str = 'season') -> pd.DataFrame:
    """Get side-by-side defensive comparison."""
    df1 = self.team1.defense.get_dataframe(window)
    df2 = self.team2.defense.get_dataframe(window)

    if df1.empty or df2.empty:
      return pd.DataFrame()

    comparison = df1[['Position', 'Avg_Per_Game']].copy()
    comparison.columns = ['Position', f'{self.team1.team_abbrev}_Allows']

    df2_avg = df2[['Position', 'Avg_Per_Game']].copy()
    df2_avg.columns = ['Position', f'{self.team2.team_abbrev}_Allows']

    comparison = comparison.merge(df2_avg, on='Position', how='outer')
    return comparison

  def get_top_scorers_comparison(self, n: int = 10) -> pd.DataFrame:
    """Get top scoreers from both teams."""
    team1_top = self.team1.get_top_scorers(n)
    team2_top = self.team2.get_top_scorers(n)

    data = []
    for player in team1_top:
      data.append({
        'Team': self.team1.team_abbrev,
        'Player': player.name,
        'Position': player.position,
        'PPG_Season': player.ppg_season,
        'PPG_L7': player.ppg_last_7,
        'Form': player.form_differential
      })

    for player in team2_top:
      data.append({
        'Team': self.team2.team_abbrev,
        'Player': player.name,
        'Position': player.position,
        'PPG_Season': player.ppg_season,
        'PPG_L7': player.ppg_last_7,
        'Form': player.form_differential
      })

    return pd.DataFrame(data)

  def get_favorable_matchups(self, min_score: float = 1.0) -> List[Tuple[str, MatchupInsight]]:
    """Get all favorable matchups across both teams"""
    favorable = []

    for insight in self.team1_insights:
      if insight.matchup_score >= min_score:
        favorable.append((self.team1.team_abbrev, insight))
    for insight in self.team2_insights:
      if insight.matchup_score >= min_score:
        favorable.append((self.team2.team_abbrev, insight))

    # Sort by matchup score
    favorable.sort(key=lambda x: x[1].matchup_score, reverse=True)
    return favorable

# ============
# DATA FETCHER
# ============

class DataFetcher:
  """Handles all NBA API calls with rate limiting."""

  def __init__(self, config: Config):
    self.config = config

  def get_team_info(self, team_abbrev: str) -> Tuple[int, str]:
    """Get team ID and name from abbreviation"""
    nba_teams = teams.get_teams()
    team = [t for t in nba_teams if t['abbreviation'] == team_abbrev.upper()]
    if not team:
      valid = [t['abbreviation'] for t in nba_teams]
      raise ValueError(f"Team '{team_abbrev}' not found. Valid: {valid}")
    return team[0]['id'], team[0]['full_name']

  def get_games(self, team_id: int, n_games: Optional[int] = None, season: str = None) -> pd.DataFrame:
    """Get games for a team."""
    season = season or self.config.current_season

    gamefinder = leaguegamefinder.LeagueGameFinder(
      team_id_nullable=team_id,
      season_nullable=season,
      season_type_nullable='Regular Season'
    )
    time.sleep(self.config.rate_limit_delay)

    games_df = gamefinder.get_data_frames()[0]
    games_df = games_df.sort_values('GAME_DATE', ascending=False)

    if n_games:
      games_df = games_df.head(n_games)

    return games_df

  def get_box_score(self, game_id: str) -> pd.DataFrame:
    """Get box score for a game"""
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    time.sleep(self.config.rate_limit_delay)
    return boxscore.get_data_frames()[0]

  def get_roster(self, team_id: int, season: str = None) -> pd.DataFrame:
    """Get team roster with positions"""
    season = season or self.config.current_season

    roster = commonteamroster.CommonTeamRoster(
      team_id=team_id,
      season=season
    )
    time.sleep(self.config.rate_limit_delay)
    return roster.get_data_frames()[0]

  def get_player_game_log(self, player_id: int, season: str = None) -> pd.DataFrame:
    """Get player game log"""
    season = season or self.config.current_season

    gamelog = playergamelog.PlayerGameLog(
      player_id=player_id,
      season=season,
      season_type_all_star='Regular Season'
    )
    time.sleep(self.config.rate_limit_delay)
    return gamelog.get_data_frames()[0]

# ================
# ANALYZER ENGINES
# ================

class DefensiveAnalyzer:
  """Analyzes defensive performance"""

  def __init__(self, data_fetcher: DataFetcher, config: Config):
    self.fetcher = data_fetcher
    self.config = config

  def analyze_defense(self, team_id: int, n_games: Optional[int] = None, season: str = None) -> List[DefensiveStats]:
    """Analyze defensive performance for a time window"""
    try:
      games = self.fetcher.get_games(team_id, n_games, season)

      if len(games) == 0:
        return []

      actual_games = len(games)
      all_opponent_data = []

      for _, game in games.iterrows():
        try:
          # Extract opponent
          matchup = game['MATCHUP']
          opponent_abbrev = matchup.replace('vs.', '@').split('@')[1].strip()
          opponent_id = self.fetcher.get_team_info(opponent_abbrev)[0]

          # Get opponent stats
          box_score = self.fetcher.get_box_score(game['GAME_ID'])
          opponent_stats = box_score[box_score['TEAM_ID'] == opponent_id]

          # Get opponent roster
          roster = self.fetcher.get_roster(opponent_id, season)
          roster_df = roster[['PLAYER_ID', 'POSITION']]

          # Merge & normalize
          merged = opponent_stats.merge(roster_df, on='PLAYER_ID', how='left')
          merged['NORM_POS'] = merged['POSITION'].apply(self.config.normalize_position)
          all_opponent_data.append(merged)

        except Exception:
          continue

      if not all_opponent_data:
        return []

      # Aggregate by position
      combined = pd.concat(all_opponent_data, ignore_index=True)
      summary = combined.groupby('NORM_POS')['PTS'].sum().reset_index()
      summary.columns = ['Position', 'Total_Points']
      summary['Avg_Per_Game'] = (summary['Total_Points'] / actual_games).round(1)
      summary['Games'] = actual_games

      # Convert to DefensiveStats objects
      stats = []
      for _, row in summary.iterrows():
        stats.append(DefensiveStats(
          position=row['Position'],
          total_points=row['Total_Points'],
          avg_per_game=row['Avg_Per_Game'],
          games_analyzed=row['Games']
        ))

      # Sort by position order
      position_map = {pos: i for i, pos in enumerate(self.config.position_order)}
      stats.sort(key=lambda x: position_map.get(x.position, 99))

      return stats

    except Exception:
      return []

class OffensiveAnalyzer:
  """Analyzes offensive performance"""

  def __init__(self, data_fetcher: DataFetcher, config: Config):
    self.fetcher = data_fetcher
    self.config = config

  def analyze_offense(self, team_id: int, season: str = None) -> List[PlayerStats]:
    """Analyze offensive performance (all players)"""
    try:
      roster = self.fetcher.get_roster(team_id, season)
      players = []

      for _, player_row in roster.iterrows():
        try:
          player_id = player_row['PLAYER_ID']

          # Get game log
          games = self.fetcher.get_player_game_log(player_id, season)

          if len(games) == 0:
            continue

          games = games.sort_values('GAME_DATE', ascending=False)
          total_games = len(games)

          # Calculate averages
          ppg_7 = games.head(7)['PTS'].mean() if total_games >= 7 else 0
          ppg_15 = games.head(15)['PTS'].mean() if total_games >= 15 else 0
          ppg_season = games['PTS'].mean()

          # Create PlayerStats object
          player = PlayerStats(
            player_id=player_id,
            name=player_row['PLAYER'],
            jersey=str(player_row['NUM']),
            position=self.config.normalize_position(player_row['POSITION']),
            listed_position=str(player_row['POSITION']),
            ppg_last_7=round(ppg_7, 1),
            ppg_last_15=round(ppg_15, 1),
            ppg_season=round(ppg_season, 1),
            games_played=total_games
          )

          players.append(player)

        except Exception:
          continue

      # Sort by position & season PPG
      position_map = {pos: i for i, pos in enumerate(self.config.position_order)}
      players.sort(key=lambda x: (position_map.get(x.position, 99), -x.ppg_season))

      return players

    except Exception:
      return []

# =====================
# MAIN MATCHUP ANALYZER
# =====================

class MatchupAnalyzer:
  """Main class for analyzing matchups between two teams."""

  def __init__(self, config: Optional[Config] = None):
    self.config = config or Config()
    self.fetcher = DataFetcher(self.config)
    self.defensive_analyzer = DefensiveAnalyzer(self.fetcher, self.config)
    self.offensive_analyzer = OffensiveAnalyzer(self.fetcher, self.config)

  def analyze_team(self, team_abbrev: str, season: str = None, verbose: bool = False) -> TeamAnalysis:
    """Analyze a single team (defense + offense)"""
    season = season or self.config.current_season

    # Get team info
    team_id, team_name = self.fetcher.get_team_info(team_abbrev)

    if verbose:
      print(f"\nAnalyzing {team_name} ({team_abbrev})...")

    # Analyze defense
    defense = DefensiveProfile()

    if verbose:
      print("  Defensive analysis...")

    defense.last_7 = self.defensive_analyzer.analyze_defense(
      team_id, n_games=7, season=season
    )
    defense.last_15 = self.defensive_analyzer.analyze_defense(
      team_id, n_games=15, season=season
    )
    defense.season = self.defensive_analyzer.analyze_defense(
      team_id, n_games=None, season=season
    )

    # Analyse offense
    if verbose:
      print("  Offensive analysis...")

    players = self.offensive_analyzer.analyze_offense(team_id, season=season)

    if verbose:
      print(f"  ✓ Complete ({len(players)} players)")

    return TeamAnalysis(
      team_id=team_id,
      team_abbrev=team_abbrev.upper(),
      team_name=team_name,
      defense=defense,
      players=players
    )

  def _identify_matchup_insights(self, offensive_team: TeamAnalysis, defensive_team: TeamAnalysis) -> List[MatchupInsight]:
    """Identify favorable matchups for one team."""
    insights = []

    # Get defensive weakness
    weak_positions = defensive_team.defense.get_weakest_positions(window='season', n=3)

    for weak_pos in weak_positions:
      # Find offensive players at this position
      attackers = offensive_team.get_players_by_position(weak_pos.position)

      if attackers:
        # Get best scorer at this position
        best_attacker = max(attackers, key=lambda p: p.ppg_season)

        # Calculate matchup score
        matchup_score = weak_pos.avg_per_game - best_attacker.ppg_season

        insights.append(MatchupInsight(
          offensive_player=best_attacker,
          defensive_position=weak_pos,
          matchup_score=matchup_score
        ))

    return insights

  def analyze_matchups(self, team1_abbrev: str, team2_abbrev: str, season: str = None, verbose: bool = True) -> MatchupAnalysis:
    """
    Analyze complete matchup between two teams.

    Args:
      team1_abbrev: First team abbreviation
      team2_abbrev: Second team abbreviation
      season: NBA season
      verbose: Print progress

    Returns:
      MatchupAnalysis object with complete data
    """
    season = season or self.config.current_season

    if verbose:
      print(f"\n{'='*70}")
      print(f"NBA MATCHUP ANALYZER: {team1_abbrev.upper()} vs {team2_abbrev.upper()}")
      print(f"Season: {season}")
      print(f"{'='*70}")

    # Analyze both teams
    team1 = self.analyze_team(team1_abbrev, season, verbose)
    team2 = self.analyze_team(team2_abbrev, season, verbose)

    # Identify matchup insights
    if verbose:
      print("\nAnalyzing matchup insights...")

    team1_insights = self._identify_matchup_insights(team1, team2)
    team2_insights = self._identify_matchup_insights(team2, team1)

    if verbose:
      print("✓ Analysis complete!\n")

    return MatchupAnalysis(
      team1=team1,
      team2=team2,
      team1_insights=team1_insights,
      team2_insights=team2_insights
    )

  # =====================
  # CONVENIENCE FUNCTIONS
  # =====================

def quick_analyze(team1: str, team2: str, season: str = None) -> MatchupAnalysis:
  """Quick analysis function for convenience"""
  analyzer = MatchupAnalyzer()
  return analyzer.analyze_matchups(team1, team2, season, verbose=True)

def get_all_teams() -> List[Dict[str, str]]:
  """Get list of all NBA teams."""
  nba_teams = teams.get_teams()
  return [{'abbreviation': t['abbreviation'], 'name': t['full_name']} for t in nba_teams]

# ======================
# COMMAND LINE INTERFACE
# ======================

if __name__ == "__main__":
  import sys

  if len(sys.argv) < 3:
    print("\nUsage: python matchup_analyzer.py TEAM1 TEAM2 [SEASON]")
    print("\nExample: python matchup_analyzer.py LAL BOS")
    sys.exit(1)

  team1 = sys.argv[1]
  team2 = sys.argv[2]
  season = sys.argv[3] if len(sys.argv) > 3 else None

  # Run analysis
  matchup = quick_analyze(team1, team2, season)

  # Display results
  print("\n" + "="*70)
  print("DEFENSIVE COMPARISON")
  print("="*70)
  print(matchup.get_defense_comparison())

  print("\n" + "="*70)
  print("TOP SCORERS")
  print("="*70)
  print(matchup.get_top_scorers_comparison())

  print("\n" + "="*70)
  print("FAVORABLE MATCHUPS")
  print("="*70)
  for team_abbrev, insight in matchup.get_favorable_matchups():
    print(f"\n{team_abbrev}: {insight.description}")
    print(f"  Matchup Score: {insight.matchup_score:+.1f}")



